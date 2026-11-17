// Package micromodel runs micromodel encoders on the CPU in pure Go. It loads any `.mmenc` file (micromodel-spec
// encoder/SPEC.md): a ModernBERT model whose linear layers are LittleBit-2 (two paths of
// diag(h) sign(U) diag(l) sign(V)ᵀ diag(g)), with 4-bit token embeddings, its tokenizer and an optional
// classification head in one file. The package ships no model: you load one.
//
//	enc, err := micromodel.Load(bytes)
//	logits, err := enc.Logits("my card got swallowed by the atm")
//
// A sign matrix times a vector uses lookup tables: for each run of 8 inputs all 256 signed sums are tabulated once,
// and each output then costs one table read per 8 inputs instead of 8 multiply-adds. An Encoder is safe for
// concurrent use.
package micromodel

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
)

type Arch struct {
	Type         string  `json:"type"`
	Hidden       int     `json:"hidden"`
	Layers       int     `json:"layers"`
	Heads        int     `json:"heads"`
	Inter        int     `json:"inter"`
	Vocab        int     `json:"vocab"`
	Eps          float64 `json:"eps"`
	Window       int     `json:"window"`
	MaxTokens    int     `json:"max_tokens"`
	GlobalLayers []int   `json:"global_layers"`
	ThetaGlobal  float64 `json:"theta_global"`
	ThetaLocal   float64 `json:"theta_local"`
}

type Head struct {
	Type   string   `json:"type"` // "classify" or "embed"
	Labels []string `json:"labels,omitempty"`
}

type Tensor struct {
	Name   string `json:"name"`
	Dtype  string `json:"dtype"`
	Shape  []int  `json:"shape"`
	Offset int    `json:"offset"`
	Bytes  int    `json:"bytes"`
}

// Info is the file header.
type Info struct {
	Format      string            `json:"format"`
	Name        string            `json:"name"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	Source      string            `json:"source"`
	Arch        Arch              `json:"arch"`
	Prefix      string            `json:"prefix"`
	Head        Head              `json:"head"`
	Linears     map[string][4]int `json:"linears"`
	Tensors     []Tensor          `json:"tensors"`
}

type path struct {
	u, vt   []byte
	h, g, l []float32
}

type linear struct {
	din, dout, r int
	paths        []path
}

type layer struct {
	attnNorm, mlpNorm []float32 // attnNorm nil: identity
	wqkv, wo, wi, wo2 linear
	global            bool
}

type Encoder struct {
	Info      Info
	Tokenizer *Tokenizer
	a         Arch
	embQ      []byte
	embS      []float32
	embNorm   []float32
	finalNorm []float32
	layers    []layer
	headW     []float32
	headB     []float32
	headScale float32
}

// Load parses a `.mmenc` file. The sign matrices and embeddings are used in place; keep b unmodified.
func Load(b []byte) (*Encoder, error) {
	if len(b) < 8 || string(b[:4]) != "MMEN" {
		return nil, errors.New("micromodel: not an .mmenc file")
	}
	hlen := int(binary.LittleEndian.Uint32(b[4:8]))
	if 8+hlen > len(b) {
		return nil, errors.New("micromodel: header past end of file")
	}
	var H Info
	if err := json.Unmarshal(bytes.TrimRight(b[8:8+hlen], " "), &H); err != nil {
		return nil, fmt.Errorf("micromodel: header: %w", err)
	}
	if H.Format != "mmenc-1" {
		return nil, fmt.Errorf("micromodel: unsupported format %q", H.Format)
	}
	if H.Arch.Type != "modernbert-littlebit2" {
		return nil, fmt.Errorf("micromodel: unsupported arch %q", H.Arch.Type)
	}
	data := b[8+hlen:]
	T := map[string]Tensor{}
	for _, t := range H.Tensors {
		T[t.Name] = t
	}
	var err error
	raw := func(n string) []byte {
		t, ok := T[n]
		switch {
		case err != nil:
		case !ok:
			err = fmt.Errorf("micromodel: missing tensor %s", n)
		case t.Offset < 0 || t.Bytes < 0 || t.Offset+t.Bytes > len(data):
			err = fmt.Errorf("micromodel: tensor %s past end of file", n)
		default:
			return data[t.Offset : t.Offset+t.Bytes]
		}
		return nil
	}
	f32 := func(n string) []float32 {
		r := raw(n)
		out := make([]float32, len(r)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(r[4*i:]))
		}
		return out
	}
	h16 := func(n string) []float32 {
		r := raw(n)
		out := make([]float32, len(r)/2)
		for i := range out {
			out[i] = f16(binary.LittleEndian.Uint16(r[2*i:]))
		}
		return out
	}
	e := &Encoder{Info: H, a: H.Arch}
	var tj TokenizerJSON
	if tb := raw("tokenizer"); err == nil {
		if err = json.Unmarshal(tb, &tj); err == nil {
			e.Tokenizer, err = newTokenizer(tj)
		}
	}
	e.embQ, e.embS = raw("tok_emb.q4"), h16("tok_emb.scale")
	e.embNorm, e.finalNorm = f32("emb_norm"), f32("final_norm")
	lin := func(n string) linear {
		s, ok := H.Linears[n]
		if !ok && err == nil {
			err = fmt.Errorf("micromodel: missing linear %s", n)
		}
		l := linear{din: s[0], dout: s[1], r: s[2]}
		for p := range s[3] {
			k := fmt.Sprintf("%s.p%d.", n, p)
			l.paths = append(l.paths, path{u: raw(k + "u"), vt: raw(k + "vt"), h: h16(k + "h"), g: h16(k + "g"), l: h16(k + "l")})
		}
		return l
	}
	for i := range H.Arch.Layers {
		p := fmt.Sprintf("l%d.", i)
		L := layer{mlpNorm: f32(p + "mlp_norm"), wqkv: lin(p + "wqkv"), wo: lin(p + "wo"), wi: lin(p + "wi"), wo2: lin(p + "wo2"),
			global: slices.Contains(H.Arch.GlobalLayers, i)}
		if _, ok := T[p+"attn_norm"]; ok {
			L.attnNorm = f32(p + "attn_norm")
		}
		e.layers = append(e.layers, L)
	}
	if H.Head.Type == "classify" {
		e.headW, e.headB = f32("head.w"), f32("head.b")
		if s := f32("head.scale"); len(s) == 1 {
			e.headScale = s[0]
		} else if err == nil {
			err = errors.New("micromodel: head.scale must have one value")
		}
	}
	if err != nil {
		return nil, err
	}
	d := e.a.Hidden
	if d%32 != 0 || e.a.Heads <= 0 || d%e.a.Heads != 0 || len(e.embQ) < e.a.Vocab*d/2 || len(e.embS) < e.a.Vocab*d/32 {
		return nil, errors.New("micromodel: arch and embedding sizes disagree")
	}
	return e, nil
}

// Labels are the classify head's labels in logit order; nil for an embedding model.
func (e *Encoder) Labels() []string { return e.Info.Head.Labels }

// IDs are the model's input ids: prefix + text, [bos] … [eos], truncated to max_tokens.
func (e *Encoder) IDs(text string) ([]int, error) {
	ids, err := e.Tokenizer.Encode(e.Info.Prefix + text)
	if max := e.a.MaxTokens; len(ids) > max {
		ids = append(ids[:max-1], e.Tokenizer.EOS)
	}
	return ids, err
}

// Embed is the L2-normalized mean of the final hidden states: the embedding, and the classifier's input.
func (e *Encoder) Embed(text string) ([]float32, error) {
	ids, err := e.IDs(text)
	if err != nil {
		return nil, err
	}
	return e.forward(ids)
}

// Logits are the classify head's logits, in Labels order.
func (e *Encoder) Logits(text string) ([]float32, error) {
	if e.Info.Head.Type != "classify" {
		return nil, fmt.Errorf("micromodel: %s is an embedding model, it has no labels", e.Info.Name)
	}
	v, err := e.Embed(text)
	if err != nil {
		return nil, err
	}
	d := e.a.Hidden
	out := make([]float32, len(e.Info.Head.Labels))
	for c := range out {
		var acc float64
		for i, x := range v {
			acc += float64(e.headW[c*d+i]) * float64(x)
		}
		out[c] = float32(float64(e.headScale) * (acc + float64(e.headB[c])))
	}
	return out, nil
}

// ---------------------------------------------------------------- math
//
// Storage is float32 and sums run in float64, as in @micromodel/cpu, so the two agree to float32 rounding.

func f16(h uint16) float32 {
	s, ex, m := float64(1), int(h>>10&0x1f), float64(h&0x3ff)
	if h&0x8000 != 0 {
		s = -1
	}
	switch ex {
	case 0:
		return float32(s * m * 0x1p-24)
	case 31:
		if m != 0 {
			return float32(math.NaN())
		}
		return float32(s * math.Inf(1))
	}
	return float32(s * (1 + m/1024) * math.Ldexp(1, ex-15))
}

func gelu(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }

func layerNorm(x []float32, T, d int, w []float32, eps float64, out []float32) {
	for t := range T {
		row, o := x[t*d:(t+1)*d], out[t*d:(t+1)*d]
		var mean, v float64
		for _, z := range row {
			mean += float64(z)
		}
		mean /= float64(d)
		for _, z := range row {
			v += (float64(z) - mean) * (float64(z) - mean)
		}
		inv := 1 / math.Sqrt(v/float64(d)+eps)
		for i, z := range row {
			o[i] = float32((float64(z) - mean) * inv * float64(w[i]))
		}
	}
}

// buildLut: lut[k*256+b] = Σ_j (bit j of b ? +z : −z)[8k+j], for each run of 8 inputs.
func buildLut(z []float32, n int, lut []float32) {
	for k := range (n + 7) >> 3 {
		o, zi := k*256, k*8
		at := func(j int) float32 {
			if zi+j < n {
				return z[zi+j]
			}
			return 0
		}
		var base float32
		for j := range 8 {
			base -= at(j)
		}
		lut[o] = base
		for b := 1; b < 256; b++ {
			low := b & -b
			j := 0
			for 1<<j != low {
				j++
			}
			lut[o+b] = lut[o+(b^low)] + 2*at(j)
		}
	}
}

// apply: Y = Σ_p h_p ⊙ sign(U_p) (l_p ⊙ sign(V_p)ᵀ (g_p ⊙ x)), for each of the T rows of X.
func (lin *linear) apply(X []float32, T int, Y []float32) {
	din, dout, r := lin.din, lin.dout, lin.r
	nbIn, nbR := (din+7)>>3, (r+7)>>3
	z, tv := make([]float32, din), make([]float32, r)
	lut := make([]float32, max(nbIn, nbR)*256)
	clear(Y[:T*dout])
	for _, p := range lin.paths {
		for t := range T {
			x := X[t*din : (t+1)*din]
			for i := range z {
				z[i] = x[i] * p.g[i]
			}
			buildLut(z, din, lut)
			for k := range r {
				row := p.vt[k*nbIn : (k+1)*nbIn]
				var acc float64
				for j, b := range row {
					acc += float64(lut[j<<8|int(b)])
				}
				tv[k] = float32(acc * float64(p.l[k]))
			}
			buildLut(tv, r, lut)
			y := Y[t*dout : (t+1)*dout]
			for o := range dout {
				row := p.u[o*nbR : (o+1)*nbR]
				var acc float64
				for j, b := range row {
					acc += float64(lut[j<<8|int(b)])
				}
				y[o] = float32(float64(y[o]) + acc*float64(p.h[o]))
			}
		}
	}
}

// ---------------------------------------------------------------- forward pass (SPEC §2–5)

func (e *Encoder) forward(ids []int) ([]float32, error) {
	a := e.a
	d, nh, inter := a.Hidden, a.Heads, a.Inter
	T, hd := len(ids), d/nh
	half := hd / 2
	x, n := make([]float32, T*d), make([]float32, T*d)
	qkv, att, proj := make([]float32, T*3*d), make([]float32, T*d), make([]float32, T*d)
	wi, act := make([]float32, T*2*inter), make([]float32, T*inter)
	for t, id := range ids {
		if id < 0 || id >= a.Vocab {
			return nil, fmt.Errorf("micromodel: token id %d out of range", id)
		}
		qo, so := id*(d>>1), id*(d>>5)
		for i := range d {
			q := e.embQ[qo+i>>1]
			if i&1 == 1 {
				q >>= 4
			}
			x[t*d+i] = float32(int(q&15)-8) * e.embS[so+i>>5]
		}
	}
	layerNorm(x, T, d, e.embNorm, a.Eps, x)
	rope := func(theta float64) (c, s []float32) {
		c, s = make([]float32, T*half), make([]float32, T*half)
		for t := range T {
			for i := range half {
				ang := float64(t) / math.Pow(theta, float64(2*i)/float64(hd))
				c[t*half+i], s[t*half+i] = float32(math.Cos(ang)), float32(math.Sin(ang))
			}
		}
		return
	}
	cg, sg := rope(a.ThetaGlobal)
	cl, sl := rope(a.ThetaLocal)
	scores, q, k := make([]float32, T), make([]float32, hd), make([]float32, T*hd)
	scale := 1 / math.Sqrt(float64(hd))
	rot := func(src []float32, cs, sn []float32, dst []float32) { // rotate-half RoPE of one head row
		for i := range half {
			x0, x1, c, s := float64(src[i]), float64(src[half+i]), float64(cs[i]), float64(sn[i])
			dst[i], dst[half+i] = float32(x0*c-x1*s), float32(x1*c+x0*s)
		}
	}
	for _, L := range e.layers {
		if L.attnNorm != nil {
			layerNorm(x, T, d, L.attnNorm, a.Eps, n)
		} else {
			copy(n, x)
		}
		L.wqkv.apply(n, T, qkv)
		c, s := cl, sl
		if L.global {
			c, s = cg, sg
		}
		for h := range nh {
			for t := range T {
				ko := t*3*d + d + h*hd
				rot(qkv[ko:], c[t*half:], s[t*half:], k[t*hd:])
			}
			for t := range T {
				rot(qkv[t*3*d+h*hd:], c[t*half:], s[t*half:], q)
				lo, hi := 0, T-1
				if !L.global {
					lo, hi = max(0, t-a.Window), min(T-1, t+a.Window)
				}
				mx := float32(math.Inf(-1))
				for j := lo; j <= hi; j++ {
					var dot float64
					for i, qi := range q {
						dot += float64(qi) * float64(k[j*hd+i])
					}
					scores[j] = float32(dot * scale)
					mx = max(mx, scores[j])
				}
				var sum float64
				for j := lo; j <= hi; j++ {
					scores[j] = float32(math.Exp(float64(scores[j] - mx)))
					sum += float64(scores[j])
				}
				ao := att[t*d+h*hd : t*d+(h+1)*hd]
				clear(ao)
				for j := lo; j <= hi; j++ {
					w, vo := float64(scores[j])/sum, j*3*d+2*d+h*hd
					for i := range ao {
						ao[i] = float32(float64(ao[i]) + w*float64(qkv[vo+i]))
					}
				}
			}
		}
		L.wo.apply(att, T, proj)
		for i := range x {
			x[i] += proj[i]
		}
		layerNorm(x, T, d, L.mlpNorm, a.Eps, n)
		L.wi.apply(n, T, wi)
		for t := range T {
			for i := range inter {
				act[t*inter+i] = float32(gelu(float64(wi[t*2*inter+i])) * float64(wi[t*2*inter+inter+i]))
			}
		}
		L.wo2.apply(act, T, proj)
		for i := range x {
			x[i] += proj[i]
		}
	}
	layerNorm(x, T, d, e.finalNorm, a.Eps, x)
	v := make([]float32, d)
	for t := range T {
		for i := range d {
			v[i] += x[t*d+i]
		}
	}
	var nrm float64
	for i := range v {
		v[i] /= float32(T)
		nrm += float64(v[i]) * float64(v[i])
	}
	if nrm = math.Sqrt(nrm); nrm == 0 {
		nrm = 1
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / nrm)
	}
	return v, nil
}
