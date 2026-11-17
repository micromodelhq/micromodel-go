// micromodel-spec encoder/conformance: tokens exactly; model ids exactly, embedding to 1e-4, logits to 1e-3 with equal argmax.
package micromodel

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
)

func fixture[T any](t *testing.T, name string) T {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var v T
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func tiny(t *testing.T) *Encoder {
	b, err := os.ReadFile("testdata/tiny.mmenc")
	if err != nil {
		t.Fatal(err)
	}
	e, err := Load(b)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func argmax[S ~[]E, E float32 | float64](a S) int {
	return slices.Index(a, slices.Max(a))
}

func TestTokens(t *testing.T) {
	e := tiny(t)
	for _, c := range fixture[[]struct {
		Text string
		IDs  []int
	}](t, "tokens.json") {
		if got, err := e.Tokenizer.Encode(c.Text); err != nil || !slices.Equal(got, c.IDs) {
			t.Errorf("%q: got %v %v, want %v", c.Text, got, err, c.IDs)
		}
	}
}

func TestLogits(t *testing.T) {
	e := tiny(t)
	for i, c := range fixture[[]struct {
		Text      string
		IDs       []int
		Embedding []float64
		Logits    []float64
	}](t, "logits.json") {
		ids, err := e.IDs(c.Text)
		if err != nil || !slices.Equal(ids, c.IDs) {
			t.Errorf("#%d ids: got %v %v, want %v", i, ids, err, c.IDs)
			continue
		}
		v, _ := e.Embed(c.Text)
		lg, _ := e.Logits(c.Text)
		for j := range v {
			if math.Abs(float64(v[j])-c.Embedding[j]) > 1e-4 {
				t.Errorf("#%d embedding[%d] = %v, want %v", i, j, v[j], c.Embedding[j])
			}
		}
		for j := range lg {
			if math.Abs(float64(lg[j])-c.Logits[j]) > 1e-3 {
				t.Errorf("#%d logits[%d] = %v, want %v", i, j, lg[j], c.Logits[j])
			}
		}
		if argmax(lg) != argmax(c.Logits) {
			t.Errorf("#%d argmax %d, want %d", i, argmax(lg), argmax(c.Logits))
		}
	}
}

func TestPretokenize(t *testing.T) {
	got := pretokenize("can't  go\n\n x1 ?? ")
	want := []string{"can", "'t", " ", " go", "\n\n", " x", "1", " ??", " "}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}
