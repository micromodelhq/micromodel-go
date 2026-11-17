package micromodel

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// TokenizerJSON is the `tokenizer` tensor (SPEC §1).
type TokenizerJSON struct {
	Type       string    `json:"type"`
	Normalizer *string   `json:"normalizer"`
	Vocab      []*string `json:"vocab"`
	Merges     []string  `json:"merges"`
	Added      []Added   `json:"added"`
	BOS        int       `json:"bos"`
	EOS        int       `json:"eos"`
}

type Added struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	LStrip  bool   `json:"lstrip"`
	RStrip  bool   `json:"rstrip"`
	Special bool   `json:"special"`
}

// Tokenizer is byte-level BPE as HF tokenizers runs it for ModernBERT: NFC, added tokens, the GPT-2
// pre-tokenizer, BPE, [bos] … [eos]. It is safe for concurrent use.
type Tokenizer struct {
	BOS, EOS int
	ids      map[string]int
	ranks    map[string]int
	added    []Added // longest content first
	nfc      bool
	bytes    [256]string

	mu    sync.Mutex
	cache map[string][]int
}

func newTokenizer(j TokenizerJSON) (*Tokenizer, error) {
	if j.Type != "bpe-bytelevel" {
		return nil, fmt.Errorf("micromodel: unsupported tokenizer %q", j.Type)
	}
	t := &Tokenizer{BOS: j.BOS, EOS: j.EOS, ids: make(map[string]int, len(j.Vocab)), ranks: make(map[string]int, len(j.Merges)),
		nfc: j.Normalizer != nil && *j.Normalizer == "NFC", cache: map[string][]int{}}
	for i, v := range j.Vocab {
		if v != nil {
			t.ids[*v] = i
		}
	}
	for i, m := range j.Merges {
		t.ranks[m] = i
	}
	t.added = slices.Clone(j.Added)
	slices.SortStableFunc(t.added, func(a, b Added) int { return len(b.Content) - len(a.Content) })
	// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to 256, 257, ...
	n := 0
	for b := range 256 {
		if b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174 {
			t.bytes[b] = string(rune(b))
		} else {
			t.bytes[b] = string(rune(256 + n))
			n++
		}
	}
	return t, nil
}

// Encode returns [bos] + tokens + [eos], with no prefix and no truncation.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t.nfc {
		text = norm.NFC.String(text)
	}
	out := []int{t.BOS}
	var err error
	start, i := 0, 0
	for i < len(text) && err == nil {
		k := slices.IndexFunc(t.added, func(a Added) bool { return strings.HasPrefix(text[i:], a.Content) })
		if k < 0 {
			i++ // byte steps are fine: a valid UTF-8 token can't match mid-rune
			continue
		}
		a, end := t.added[k], i
		for a.LStrip && end > start {
			r, n := utf8.DecodeLastRuneInString(text[:end])
			if !unicode.IsSpace(r) {
				break
			}
			end -= n
		}
		if end > start {
			out, err = t.plain(text[start:end], out)
		}
		out = append(out, a.ID)
		i += len(a.Content)
		for a.RStrip && i < len(text) {
			r, n := utf8.DecodeRuneInString(text[i:])
			if !unicode.IsSpace(r) {
				break
			}
			i += n
		}
		start = i
	}
	if err == nil && start < len(text) {
		out, err = t.plain(text[start:], out)
	}
	return append(out, t.EOS), err
}

func (t *Tokenizer) plain(text string, out []int) ([]int, error) {
	for _, w := range pretokenize(text) {
		var s strings.Builder
		for i := range len(w) {
			s.WriteString(t.bytes[w[i]])
		}
		ids, err := t.bpe(s.String())
		if err != nil {
			return out, err
		}
		out = append(out, ids...)
	}
	return out, nil
}

func (t *Tokenizer) bpe(word string) ([]int, error) {
	t.mu.Lock()
	hit, ok := t.cache[word]
	t.mu.Unlock()
	if ok {
		return hit, nil
	}
	var parts []string
	for _, r := range word {
		parts = append(parts, string(r))
	}
	for len(parts) > 1 {
		best, rank := -1, 0
		for i := 0; i < len(parts)-1; i++ {
			if r, ok := t.ranks[parts[i]+" "+parts[i+1]]; ok && (best < 0 || r < rank) {
				best, rank = i, r
			}
		}
		if best < 0 {
			break
		}
		a, b := parts[best], parts[best+1]
		next := parts[:0:0]
		for i := 0; i < len(parts); i++ {
			if i < len(parts)-1 && parts[i] == a && parts[i+1] == b {
				next = append(next, a+b)
				i++
			} else {
				next = append(next, parts[i])
			}
		}
		parts = next
	}
	out := make([]int, len(parts))
	for i, p := range parts {
		id, ok := t.ids[p]
		if !ok {
			return nil, fmt.Errorf("micromodel: symbol %q not in vocab", p)
		}
		out[i] = id
	}
	t.mu.Lock()
	if len(t.cache) < 50_000 {
		t.cache[word] = out
	}
	t.mu.Unlock()
	return out, nil
}

// pretokenize splits text like the GPT-2 regex, which RE2 can't express (it has a lookahead):
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func pretokenize(s string) []string {
	var out []string
	other := func(r rune) bool { return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) }
	// run returns the end of the longest run of class starting at i (i if none).
	run := func(i int, class func(rune) bool) int {
		for i < len(s) {
			r, n := utf8.DecodeRuneInString(s[i:])
			if !class(r) {
				break
			}
			i += n
		}
		return i
	}
	for i := 0; i < len(s); {
		end := i
		if s[i] == '\'' {
			for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
				if strings.HasPrefix(s[i+1:], c) {
					end = i + 1 + len(c)
					break
				}
			}
		}
		for _, class := range []func(rune) bool{unicode.IsLetter, unicode.IsNumber, other} {
			if end > i {
				break
			}
			j := i
			if s[j] == ' ' {
				j++
			}
			if e := run(j, class); e > j {
				end = e
			}
		}
		if end == i { // whitespace: \s+(?!\S) leaves the last space for the next word, else \s+
			end = run(i, unicode.IsSpace)
			if end < len(s) {
				_, n := utf8.DecodeLastRuneInString(s[:end])
				if end-n > i {
					end -= n
				}
			}
		}
		if end == i { // unreachable for valid UTF-8; never loop
			_, n := utf8.DecodeRuneInString(s[i:])
			end = i + n
		}
		out = append(out, s[i:end])
		i = end
	}
	return out
}
