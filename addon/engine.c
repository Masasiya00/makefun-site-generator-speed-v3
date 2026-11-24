/* site-generator-speed-01 v3 engine: a from-scratch, byte-exact reimplementation
 * of makefun-site 1.0.0 in C, exposed to Node through N-API.
 *
 * The whole pipeline (JSON parsing, front matter, Markdown, templates,
 * collections, serialisation) runs here. Strings are byte slices; the input
 * JSON text is UTF-8, and lone surrogates arriving through \u escapes are
 * kept in WTF-8 form (a 3-byte ED A0..ED BF sequence) so every operation
 * that JavaScript performs on UTF-16 code units stays exact.
 *
 * Memory: one bump arena per case holds every string and node; it is thrown
 * away whole when the case is done. Small growable buffers (Buf) are used
 * only while building output text.
 */

#define NAPI_VERSION 8
#include <node_api.h>

#include <fcntl.h>
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ arena */

/* A bump arena made of chained blocks. Pointers into it never move: when
 * the current block fills up a fresh block is chained on, so a Str taken
 * earlier stays valid no matter how much is allocated afterwards. */
typedef struct ABlock {
  struct ABlock *next;
  size_t cap, len;
  unsigned char data[];
} ABlock;

typedef struct {
  ABlock *head, *tail;
  size_t blk_cap;
} Arena;

static void *ar_alloc(Arena *a, size_t n) {
  n = (n + 7) & ~(size_t)7;
  if (!a->tail || a->tail->len + n > a->tail->cap) {
    size_t nc = a->blk_cap ? a->blk_cap * 2 : (size_t)1 << 20;
    if (nc < n) nc = n;
    if (nc > ((size_t)1 << 26) && n <= ((size_t)1 << 26)) nc = (size_t)1 << 26;
    ABlock *b = (ABlock *)malloc(sizeof(ABlock) + nc);
    if (!b) abort();
    b->next = NULL;
    b->cap = nc;
    b->len = 0;
    if (a->tail) a->tail->next = b;
    else a->head = b;
    a->tail = b;
    a->blk_cap = nc;
  }
  void *r = a->tail->data + a->tail->len;
  a->tail->len += n;
  return r;
}

/* Allocate a fresh block and copy `old` (oldn bytes) into it; the old block
 * stays valid inside the arena, which is fine: the arena dies per case. */
static void *ar_grow_copy(Arena *a, const void *old, size_t oldn, size_t newn) {
  void *r = ar_alloc(a, newn);
  if (oldn) memcpy(r, old, oldn);
  return r;
}

static void arena_free(Arena *a) {
  ABlock *b = a->head;
  while (b) {
    ABlock *nx = b->next;
    free(b);
    b = nx;
  }
  a->head = a->tail = NULL;
  a->blk_cap = 0;
}

/* -------------------------------------------------------------------- Buf */

typedef struct {
  unsigned char *b;
  size_t len, cap;
} Buf;

static void buf_reserve(Buf *u, size_t need) {
  if (u->len + need <= u->cap) return;
  size_t nc = u->cap ? u->cap : 256;
  while (nc < u->len + need) nc *= 2;
  u->b = (unsigned char *)realloc(u->b, nc);
  if (!u->b) abort();
  u->cap = nc;
}

static void buf_addn(Buf *u, const void *p, size_t n) {
  if (!n) return;
  buf_reserve(u, n);
  memcpy(u->b + u->len, p, n);
  u->len += n;
}

static void buf_addc(Buf *u, unsigned char c) {
  buf_reserve(u, 1);
  u->b[u->len++] = c;
}

static void buf_adds(Buf *u, const char *s) { buf_addn(u, s, strlen(s)); }

static void buf_free(Buf *u) {
  free(u->b);
  u->b = NULL;
  u->len = u->cap = 0;
}

/* -------------------------------------------------------------------- Str */

typedef struct {
  const unsigned char *p;
  size_t n;
} Str;

static Str str_from_c(const char *s) {
  Str r;
  r.p = (const unsigned char *)s;
  r.n = strlen(s);
  return r;
}

static int str_eq(Str a, Str b) {
  return a.n == b.n && (a.n == 0 || memcmp(a.p, b.p, a.n) == 0);
}

static int str_eq_c(Str a, const char *s) {
  size_t n = strlen(s);
  return a.n == n && (n == 0 || memcmp(a.p, s, n) == 0);
}

static int str_starts(Str a, const char *pre) {
  size_t n = strlen(pre);
  return a.n >= n && memcmp(a.p, pre, n) == 0;
}

static int str_ends(Str a, const char *suf) {
  size_t n = strlen(suf);
  return a.n >= n && memcmp(a.p + a.n - n, suf, n) == 0;
}

static Str ar_str(Arena *a, const unsigned char *p, size_t n) {
  Str r;
  unsigned char *d = (unsigned char *)ar_alloc(a, n ? n : 1);
  if (n) memcpy(d, p, n);
  r.p = d;
  r.n = n;
  return r;
}

static Str buf_to_str(Arena *a, const Buf *u) { return ar_str(a, u->b, u->len); }

/* ------------------------------------------------------- UTF-8 primitives */

/* Decode one UTF-8 sequence (WTF-8 surrogates included: ED A0..ED BF decode
 * to 0xD800..0xDFFF). Returns bytes consumed, or -1 when invalid. */
static int u8dec(const unsigned char *p, const unsigned char *e, uint32_t *cp) {
  if (p >= e) return 0;
  unsigned char c = p[0];
  if (c < 0x80) {
    *cp = c;
    return 1;
  }
  if (c < 0xC2) return -1;
  if (c < 0xE0) {
    if (e - p < 2 || (p[1] & 0xC0) != 0x80) return -1;
    *cp = ((uint32_t)(c & 0x1F) << 6) | (p[1] & 0x3F);
    return 2;
  }
  if (c < 0xF0) {
    if (e - p < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return -1;
    uint32_t v = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    if (v < 0x800) return -1;
    *cp = v;
    return 3;
  }
  if (c < 0xF5) {
    if (e - p < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return -1;
    uint32_t v = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
                 ((uint32_t)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    if (v < 0x10000 || v > 0x10FFFF) return -1;
    *cp = v;
    return 4;
  }
  return -1;
}

static int u8enc(uint32_t cp, unsigned char *out) {
  if (cp < 0x80) {
    out[0] = (unsigned char)cp;
    return 1;
  }
  if (cp < 0x800) {
    out[0] = (unsigned char)(0xC0 | (cp >> 6));
    out[1] = (unsigned char)(0x80 | (cp & 63));
    return 2;
  }
  if (cp < 0x10000) {
    out[0] = (unsigned char)(0xE0 | (cp >> 12));
    out[1] = (unsigned char)(0x80 | ((cp >> 6) & 63));
    out[2] = (unsigned char)(0x80 | (cp & 63));
    return 3;
  }
  out[0] = (unsigned char)(0xF0 | (cp >> 18));
  out[1] = (unsigned char)(0x80 | ((cp >> 12) & 63));
  out[2] = (unsigned char)(0x80 | ((cp >> 6) & 63));
  out[3] = (unsigned char)(0x80 | (cp & 63));
  return 4;
}

/* JavaScript \s: White_Space plus the line terminators. */
static int js_space(uint32_t cp) {
  if (cp >= 0x09 && cp <= 0x0D) return 1;
  if (cp == 0x20 || cp == 0xA0 || cp == 0x1680 || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
      cp == 0x205F || cp == 0x3000 || cp == 0xFEFF)
    return 1;
  if (cp >= 0x2000 && cp <= 0x200A) return 1;
  return 0;
}

/* Whether the code point at `p` (inside [p, e)) is JavaScript whitespace. */
static int at_space(const unsigned char *p, const unsigned char *e) {
  if (p >= e) return 0;
  if (*p < 0x80) return js_space(*p);
  uint32_t cp;
  if (u8dec(p, e, &cp) < 0) return 0;
  return js_space(cp);
}

/* Trim JavaScript whitespace from both ends of a slice. */
static Str trim_str(Str s) {
  const unsigned char *p = s.p, *e = s.p + s.n;
  while (p < e && at_space(p, e)) {
    uint32_t cp;
    int n = u8dec(p, e, &cp);
    p += n > 0 ? n : 1;
  }
  while (e > p) {
    /* step back one code point: rewind over continuation bytes */
    const unsigned char *q = e - 1;
    while (q > p && (*q & 0xC0) == 0x80) q--;
    uint32_t cp;
    if (u8dec(q, e, &cp) < 0 || !js_space(cp)) break;
    e = q;
  }
  Str r = {p, (size_t)(e - p)};
  return r;
}

/* UTF-16 code-unit iterator over a UTF-8/WTF-8 slice. */
typedef struct {
  const unsigned char *p, *e;
  uint32_t pend;
  int has;
} U16;

static int u16_next(U16 *it) {
  if (it->has) {
    it->has = 0;
    return (int)it->pend;
  }
  if (it->p >= it->e) return -1;
  uint32_t cp;
  int n = u8dec(it->p, it->e, &cp);
  if (n < 0) {
    int v = it->p[0];
    it->p++;
    return v;
  }
  it->p += n;
  if (cp >= 0x10000) {
    it->pend = 0xDC00 + (cp & 0x3FF);
    it->has = 1;
    return (int)(0xD800 + ((cp - 0x10000) >> 10));
  }
  return (int)cp;
}

/* Code-unit order, exactly JavaScript's `<` on strings. */
static int str_cmp_u16(Str a, Str b) {
  U16 x = {a.p, a.p + a.n, 0, 0}, y = {b.p, b.p + b.n, 0, 0};
  for (;;) {
    int ca = u16_next(&x), cb = u16_next(&y);
    if (ca < 0 && cb < 0) return 0;
    if (ca < 0) return -1;
    if (cb < 0) return 1;
    if (ca != cb) return ca < cb ? -1 : 1;
  }
}

/* ------------------------------------------------------------ string map */

typedef struct {
  Str key;
  int val;
  unsigned char used;
} MEnt;

typedef struct {
  MEnt *s;
  size_t cap, len;
} SMap;

static uint64_t str_hash(Str k) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < k.n; i++) {
    h ^= k.p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static void smap_init(SMap *m, size_t cap) {
  m->cap = cap ? cap : 64;
  m->len = 0;
  m->s = (MEnt *)calloc(m->cap, sizeof(MEnt));
  if (!m->s) abort();
}

static void smap_free(SMap *m) {
  free(m->s);
  m->s = NULL;
  m->cap = m->len = 0;
}

static void smap_grow(SMap *m) {
  size_t nc = m->cap * 2;
  MEnt *ns = (MEnt *)calloc(nc, sizeof(MEnt));
  if (!ns) abort();
  for (size_t i = 0; i < m->cap; i++) {
    if (!m->s[i].used) continue;
    size_t j = str_hash(m->s[i].key) & (nc - 1);
    while (ns[j].used) j = (j + 1) & (nc - 1);
    ns[j] = m->s[i];
  }
  free(m->s);
  m->s = ns;
  m->cap = nc;
}

static int smap_get(SMap *m, Str key) {
  if (!m->s) return -1;
  size_t j = str_hash(key) & (m->cap - 1);
  while (m->s[j].used) {
    if (str_eq(m->s[j].key, key)) return m->s[j].val;
    j = (j + 1) & (m->cap - 1);
  }
  return -1;
}

static int smap_has(SMap *m, Str key) { return smap_get(m, key) >= 0; }

/* Insert or overwrite; returns the previous value or -1. */
static int smap_put(SMap *m, Str key, int val) {
  if (!m->s) smap_init(m, 64);
  if ((m->len + 1) * 10 > m->cap * 7) smap_grow(m);
  size_t j = str_hash(key) & (m->cap - 1);
  while (m->s[j].used) {
    if (str_eq(m->s[j].key, key)) {
      int old = m->s[j].val;
      m->s[j].val = val;
      return old;
    }
    j = (j + 1) & (m->cap - 1);
  }
  m->s[j].used = 1;
  m->s[j].key = key;
  m->s[j].val = val;
  m->len++;
  return -1;
}

/* ------------------------------------------------------- stable merge sort */

/* Sort `n` elements of `sz` bytes in place, stable, by `cmp` (which takes
 * pointers to two elements). `tmp` must hold n elements. */
static void msort_gen(void *base, size_t n, size_t sz, void *tmp, int (*cmp)(const void *, const void *)) {
  if (n < 2) return;
  size_t h = n / 2;
  msort_gen(base, h, sz, tmp, cmp);
  msort_gen((char *)base + h * sz, n - h, sz, (char *)tmp + h * sz, cmp);
  char *a = (char *)base, *t = (char *)tmp;
  size_t i = 0, j = h, k = 0;
  while (i < h && j < n) {
    if (cmp(a + i * sz, a + j * sz) <= 0) {
      memcpy(t + k * sz, a + i * sz, sz);
      i++;
    } else {
      memcpy(t + k * sz, a + j * sz, sz);
      j++;
    }
    k++;
  }
  while (i < h) {
    memcpy(t + k * sz, a + i * sz, sz);
    i++;
    k++;
  }
  while (j < n) {
    memcpy(t + k * sz, a + j * sz, sz);
    j++;
    k++;
  }
  memcpy(a, t, n * sz);
}

/* ---------------------------------------------------------- JSON parsing */

typedef struct {
  Str title, base;
  long long per_page, recent;
} Site;

typedef struct {
  Str name, src;
} Tmpl;

typedef struct {
  Str path, content;
} FileEnt;

typedef struct {
  Site site;
  Tmpl *tmpls;
  int ntmpl, captmpl;
  FileEnt *files;
  int nfile, capfile;
} CaseIn;

typedef struct {
  const unsigned char *p, *e;
  Arena *ar;
  Buf scratch; /* reused while unescaping one string */
  int err;
} JP;

static void jp_ws(JP *j) {
  while (j->p < j->e && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

/* A JSON string value. Zero-copy when it holds no backslash. */
static Str jp_string(JP *j) {
  Str empty = {j->p, 0};
  if (j->err || j->p >= j->e || *j->p != '"') {
    j->err = 1;
    return empty;
  }
  const unsigned char *s = j->p + 1;
  const unsigned char *rq = (const unsigned char *)memchr(s, '"', (size_t)(j->e - s));
  const unsigned char *rb = (const unsigned char *)memchr(s, '\\', (size_t)(j->e - s));
  const unsigned char *q = rq && rb ? (rq < rb ? rq : rb) : (rq ? rq : rb);
  if (!q) q = j->e;
  if (q < j->e && *q == '"') {
    Str r = {s, (size_t)(q - s)};
    j->p = q + 1;
    return r;
  }
  if (q >= j->e) {
    j->err = 1;
    return empty;
  }
  /* Slow path: escapes present. First copy the plain prefix before the
   * first backslash, then walk the rest. */
  Buf *u = &j->scratch;
  u->len = 0;
  buf_addn(u, s, (size_t)(q - s));
  while (q < j->e) {
    unsigned char c = *q;
    if (c == '"') {
      j->p = q + 1;
      return buf_to_str(j->ar, u);
    }
    if (c != '\\') {
      const unsigned char *rq2 = (const unsigned char *)memchr(q, '"', (size_t)(j->e - q));
      const unsigned char *rb2 = (const unsigned char *)memchr(q, '\\', (size_t)(j->e - q));
      const unsigned char *r = rq2 && rb2 ? (rq2 < rb2 ? rq2 : rb2) : (rq2 ? rq2 : rb2);
      if (!r) r = j->e;
      buf_addn(u, q, (size_t)(r - q));
      q = r;
      continue;
    }
    q++;
    if (q >= j->e) break;
    unsigned char n = *q++;
    switch (n) {
      case '"': buf_addc(u, '"'); break;
      case '\\': buf_addc(u, '\\'); break;
      case '/': buf_addc(u, '/'); break;
      case 'b': buf_addc(u, '\b'); break;
      case 'f': buf_addc(u, '\f'); break;
      case 'n': buf_addc(u, '\n'); break;
      case 'r': buf_addc(u, '\r'); break;
      case 't': buf_addc(u, '\t'); break;
      case 'u': {
        /* Read four hex digits at q (q points past the 'u'). */
        uint32_t cp = 0;
        int ok = 1;
        if (j->e - q < 4) ok = 0;
        for (int h = 0; ok && h < 4; h++) {
          unsigned char c = q[h];
          cp <<= 4;
          if (c >= '0' && c <= '9') cp |= (uint32_t)(c - '0');
          else if (c >= 'a' && c <= 'f') cp |= (uint32_t)(c - 'a' + 10);
          else if (c >= 'A' && c <= 'F') cp |= (uint32_t)(c - 'A' + 10);
          else ok = 0;
        }
        if (!ok) {
          j->err = 1;
          return empty;
        }
        q += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && q + 1 < j->e && q[0] == '\\' && q[1] == 'u') {
          uint32_t lo = 0;
          int ok2 = 1;
          if (j->e - (q + 2) < 4) ok2 = 0;
          for (int h = 0; ok2 && h < 4; h++) {
            unsigned char c = q[2 + h];
            lo <<= 4;
            if (c >= '0' && c <= '9') lo |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') lo |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') lo |= (uint32_t)(c - 'A' + 10);
            else ok2 = 0;
          }
          if (ok2 && lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            q += 6;
          }
        }
        unsigned char enc[4];
        int m = u8enc(cp, enc);
        buf_addn(u, enc, (size_t)m);
        break;
      }
      default:
        j->err = 1;
        return empty;
    }
  }
  j->err = 1;
  return empty;
}

static long long jp_number(JP *j) {
  const unsigned char *s = j->p;
  int isint = 1;
  if (j->p < j->e && *j->p == '-') j->p++;
  while (j->p < j->e && *j->p >= '0' && *j->p <= '9') j->p++;
  if (j->p < j->e && *j->p == '.') {
    isint = 0;
    j->p++;
    while (j->p < j->e && *j->p >= '0' && *j->p <= '9') j->p++;
  }
  if (j->p < j->e && (*j->p == 'e' || *j->p == 'E')) {
    isint = 0;
    j->p++;
    if (j->p < j->e && (*j->p == '+' || *j->p == '-')) j->p++;
    while (j->p < j->e && *j->p >= '0' && *j->p <= '9') j->p++;
  }
  size_t n = (size_t)(j->p - s);
  char tmp[40];
  if (n >= sizeof(tmp)) {
    j->err = 1;
    return 0;
  }
  memcpy(tmp, s, n);
  tmp[n] = 0;
  if (isint) return strtoll(tmp, NULL, 10);
  return (long long)strtod(tmp, NULL);
}

static void jp_skip_value(JP *j);

static void jp_skip_value(JP *j) {
  jp_ws(j);
  if (j->p >= j->e) {
    j->err = 1;
    return;
  }
  unsigned char c = *j->p;
  if (c == '"') {
    (void)jp_string(j);
    return;
  }
  if (c == '{') {
    j->p++;
    jp_ws(j);
    if (j->p < j->e && *j->p == '}') {
      j->p++;
      return;
    }
    for (;;) {
      jp_ws(j);
      (void)jp_string(j);
      jp_ws(j);
      if (j->p >= j->e || *j->p != ':') {
        j->err = 1;
        return;
      }
      j->p++;
      jp_skip_value(j);
      if (j->err) return;
      jp_ws(j);
      if (j->p < j->e && *j->p == ',') {
        j->p++;
        continue;
      }
      if (j->p < j->e && *j->p == '}') {
        j->p++;
        return;
      }
      j->err = 1;
      return;
    }
  }
  if (c == '[') {
    j->p++;
    jp_ws(j);
    if (j->p < j->e && *j->p == ']') {
      j->p++;
      return;
    }
    for (;;) {
      jp_skip_value(j);
      if (j->err) return;
      jp_ws(j);
      if (j->p < j->e && *j->p == ',') {
        j->p++;
        continue;
      }
      if (j->p < j->e && *j->p == ']') {
        j->p++;
        return;
      }
      j->err = 1;
      return;
    }
  }
  if (c == 't') {
    if (j->e - j->p >= 4 && memcmp(j->p, "true", 4) == 0) j->p += 4;
    else j->err = 1;
    return;
  }
  if (c == 'f') {
    if (j->e - j->p >= 5 && memcmp(j->p, "false", 5) == 0) j->p += 5;
    else j->err = 1;
    return;
  }
  if (c == 'n') {
    if (j->e - j->p >= 4 && memcmp(j->p, "null", 4) == 0) j->p += 4;
    else j->err = 1;
    return;
  }
  if (c == '-' || (c >= '0' && c <= '9')) {
    (void)jp_number(j);
    return;
  }
  j->err = 1;
}

/* Parse one case's input object (the parser sits on its '{'). */
static void jp_parse_input(JP *j, CaseIn *ci) {
  memset(ci, 0, sizeof(*ci));
  ci->site.per_page = 0;
  ci->site.recent = 0;
  jp_ws(j);
  if (j->p >= j->e || *j->p != '{') {
    j->err = 1;
    return;
  }
  j->p++;
  jp_ws(j);
  if (j->p < j->e && *j->p == '}') {
    j->p++;
    return;
  }
  for (;;) {
    jp_ws(j);
    Str key = jp_string(j);
    if (j->err) return;
    jp_ws(j);
    if (j->p >= j->e || *j->p != ':') {
      j->err = 1;
      return;
    }
    j->p++;
    jp_ws(j);
    if (str_eq_c(key, "site")) {
      if (j->p >= j->e || *j->p != '{') {
        j->err = 1;
        return;
      }
      j->p++;
      jp_ws(j);
      if (j->p < j->e && *j->p == '}') {
        j->p++;
      } else {
        for (;;) {
          jp_ws(j);
          Str sk = jp_string(j);
          if (j->err) return;
          jp_ws(j);
          if (j->p >= j->e || *j->p != ':') {
            j->err = 1;
            return;
          }
          j->p++;
          jp_ws(j);
          if (str_eq_c(sk, "title")) ci->site.title = jp_string(j);
          else if (str_eq_c(sk, "base_url")) ci->site.base = jp_string(j);
          else if (str_eq_c(sk, "per_page")) ci->site.per_page = jp_number(j);
          else if (str_eq_c(sk, "recent")) ci->site.recent = jp_number(j);
          else jp_skip_value(j);
          if (j->err) return;
          jp_ws(j);
          if (j->p < j->e && *j->p == ',') {
            j->p++;
            continue;
          }
          if (j->p < j->e && *j->p == '}') {
            j->p++;
            break;
          }
          j->err = 1;
          return;
        }
      }
    } else if (str_eq_c(key, "templates")) {
      if (j->p >= j->e || *j->p != '{') {
        j->err = 1;
        return;
      }
      j->p++;
      jp_ws(j);
      if (j->p < j->e && *j->p == '}') {
        j->p++;
      } else {
        for (;;) {
          jp_ws(j);
          Str name = jp_string(j);
          if (j->err) return;
          jp_ws(j);
          if (j->p >= j->e || *j->p != ':') {
            j->err = 1;
            return;
          }
          j->p++;
          jp_ws(j);
          Str src = jp_string(j);
          if (j->err) return;
          if (ci->ntmpl == ci->captmpl) {
            int nc = ci->captmpl ? ci->captmpl * 2 : 16;
            ci->tmpls = (Tmpl *)ar_grow_copy(j->ar, ci->tmpls, (size_t)ci->ntmpl * sizeof(Tmpl),
                                            (size_t)nc * sizeof(Tmpl));
            ci->captmpl = nc;
          }
          ci->tmpls[ci->ntmpl].name = name;
          ci->tmpls[ci->ntmpl].src = src;
          ci->ntmpl++;
          jp_ws(j);
          if (j->p < j->e && *j->p == ',') {
            j->p++;
            continue;
          }
          if (j->p < j->e && *j->p == '}') {
            j->p++;
            break;
          }
          j->err = 1;
          return;
        }
      }
    } else if (str_eq_c(key, "files")) {
      if (j->p >= j->e || *j->p != '[') {
        j->err = 1;
        return;
      }
      j->p++;
      jp_ws(j);
      if (j->p < j->e && *j->p == ']') {
        j->p++;
      } else {
        for (;;) {
          jp_ws(j);
          if (j->p >= j->e || *j->p != '{') {
            j->err = 1;
            return;
          }
          j->p++;
          Str path = {j->p, 0}, content = {j->p, 0};
          jp_ws(j);
          if (j->p < j->e && *j->p == '}') {
            j->p++;
          } else {
            for (;;) {
              jp_ws(j);
              Str fk = jp_string(j);
              if (j->err) return;
              jp_ws(j);
              if (j->p >= j->e || *j->p != ':') {
                j->err = 1;
                return;
              }
              j->p++;
              jp_ws(j);
              if (str_eq_c(fk, "path")) path = jp_string(j);
              else if (str_eq_c(fk, "content")) content = jp_string(j);
              else jp_skip_value(j);
              if (j->err) return;
              jp_ws(j);
              if (j->p < j->e && *j->p == ',') {
                j->p++;
                continue;
              }
              if (j->p < j->e && *j->p == '}') {
                j->p++;
                break;
              }
              j->err = 1;
              return;
            }
          }
          if (ci->nfile == ci->capfile) {
            int nc = ci->capfile ? ci->capfile * 2 : 64;
            ci->files = (FileEnt *)ar_grow_copy(j->ar, ci->files, (size_t)ci->nfile * sizeof(FileEnt),
                                                (size_t)nc * sizeof(FileEnt));
            ci->capfile = nc;
          }
          ci->files[ci->nfile].path = path;
          ci->files[ci->nfile].content = content;
          ci->nfile++;
          jp_ws(j);
          if (j->p < j->e && *j->p == ',') {
            j->p++;
            continue;
          }
          if (j->p < j->e && *j->p == ']') {
            j->p++;
            break;
          }
          j->err = 1;
          return;
        }
      }
    } else {
      jp_skip_value(j);
    }
    if (j->err) return;
    jp_ws(j);
    if (j->p < j->e && *j->p == ',') {
      j->p++;
      continue;
    }
    if (j->p < j->e && *j->p == '}') {
      j->p++;
      return;
    }
    j->err = 1;
    return;
  }
}

/* Parse the top-level cases array [{id, input}, ...]. */
static CaseIn *jp_parse_cases(JP *j, int *out_n) {
  *out_n = 0;
  jp_ws(j);
  if (j->p >= j->e || *j->p != '[') {
    j->err = 1;
    return NULL;
  }
  j->p++;
  CaseIn *cases = NULL;
  int n = 0, cap = 0;
  jp_ws(j);
  if (j->p < j->e && *j->p == ']') {
    j->p++;
    return NULL;
  }
  for (;;) {
    jp_ws(j);
    if (j->p >= j->e || *j->p != '{') {
      j->err = 1;
      return NULL;
    }
    j->p++;
    CaseIn ci;
    int have_input = 0;
    jp_ws(j);
    if (j->p < j->e && *j->p == '}') {
      j->p++;
    } else {
      for (;;) {
        jp_ws(j);
        Str key = jp_string(j);
        if (j->err) return NULL;
        jp_ws(j);
        if (j->p >= j->e || *j->p != ':') {
          j->err = 1;
          return NULL;
        }
        j->p++;
        jp_ws(j);
        if (str_eq_c(key, "input")) {
          jp_parse_input(j, &ci);
          have_input = 1;
        } else {
          jp_skip_value(j);
        }
        if (j->err) return NULL;
        jp_ws(j);
        if (j->p < j->e && *j->p == ',') {
          j->p++;
          continue;
        }
        if (j->p < j->e && *j->p == '}') {
          j->p++;
          break;
        }
        j->err = 1;
        return NULL;
      }
    }
    if (!have_input) {
      j->err = 1;
      return NULL;
    }
    if (n == cap) {
      int nc = cap ? cap * 2 : 4;
      cases = (CaseIn *)ar_grow_copy(j->ar, cases, (size_t)n * sizeof(CaseIn), (size_t)nc * sizeof(CaseIn));
      cap = nc;
    }
    cases[n++] = ci;
    jp_ws(j);
    if (j->p < j->e && *j->p == ',') {
      j->p++;
      continue;
    }
    if (j->p < j->e && *j->p == ']') {
      j->p++;
      break;
    }
    j->err = 1;
    return NULL;
  }
  *out_n = n;
  return cases;
}

/* ------------------------------------------------- NFKD tables (hand-typed) */

/* Compatibility decompositions that can yield ASCII. A code point whose
 * decomposition yields only non-ASCII letters is left out on purpose: it
 * behaves exactly like a gap either way. Table is sorted by cp. */
typedef struct {
  uint32_t cp;
  uint8_t n;
  uint32_t d[4];
} Decomp;

#define D1(c, a) {(c), 1, {(a), 0, 0, 0}}
#define D2(c, a, b) {(c), 2, {(a), (b), 0, 0}}
#define D3(c, a, b, d) {(c), 3, {(a), (b), (d), 0}}
#define D4(c, a, b, d, e) {(c), 4, {(a), (b), (d), (e)}}

static const Decomp DECOMP[] = {
    /* Latin-1 supplement */
    D1(0x00AA, 'a'),
    D1(0x00B2, '2'),
    D1(0x00B3, '3'),
    D1(0x00B9, '1'),
    D1(0x00BA, 'o'),
    D3(0x00BC, '1', 0x2044, '4'),
    D3(0x00BD, '1', 0x2044, '2'),
    D3(0x00BE, '3', 0x2044, '4'),
    D2(0x00C0, 'A', 0x0300),
    D2(0x00C1, 'A', 0x0301),
    D2(0x00C2, 'A', 0x0302),
    D2(0x00C3, 'A', 0x0303),
    D2(0x00C4, 'A', 0x0308),
    D2(0x00C5, 'A', 0x030A),
    D2(0x00C7, 'C', 0x0327),
    D2(0x00C8, 'E', 0x0300),
    D2(0x00C9, 'E', 0x0301),
    D2(0x00CA, 'E', 0x0302),
    D2(0x00CB, 'E', 0x0308),
    D2(0x00CC, 'I', 0x0300),
    D2(0x00CD, 'I', 0x0301),
    D2(0x00CE, 'I', 0x0302),
    D2(0x00CF, 'I', 0x0308),
    D2(0x00D1, 'N', 0x0303),
    D2(0x00D2, 'O', 0x0300),
    D2(0x00D3, 'O', 0x0301),
    D2(0x00D4, 'O', 0x0302),
    D2(0x00D5, 'O', 0x0303),
    D2(0x00D6, 'O', 0x0308),
    D2(0x00D9, 'U', 0x0300),
    D2(0x00DA, 'U', 0x0301),
    D2(0x00DB, 'U', 0x0302),
    D2(0x00DC, 'U', 0x0308),
    D2(0x00DD, 'Y', 0x0301),
    D2(0x00E0, 'a', 0x0300),
    D2(0x00E1, 'a', 0x0301),
    D2(0x00E2, 'a', 0x0302),
    D2(0x00E3, 'a', 0x0303),
    D2(0x00E4, 'a', 0x0308),
    D2(0x00E5, 'a', 0x030A),
    D2(0x00E7, 'c', 0x0327),
    D2(0x00E8, 'e', 0x0300),
    D2(0x00E9, 'e', 0x0301),
    D2(0x00EA, 'e', 0x0302),
    D2(0x00EB, 'e', 0x0308),
    D2(0x00EC, 'i', 0x0300),
    D2(0x00ED, 'i', 0x0301),
    D2(0x00EE, 'i', 0x0302),
    D2(0x00EF, 'i', 0x0308),
    D2(0x00F1, 'n', 0x0303),
    D2(0x00F2, 'o', 0x0300),
    D2(0x00F3, 'o', 0x0301),
    D2(0x00F4, 'o', 0x0302),
    D2(0x00F5, 'o', 0x0303),
    D2(0x00F6, 'o', 0x0308),
    D2(0x00F9, 'u', 0x0300),
    D2(0x00FA, 'u', 0x0301),
    D2(0x00FB, 'u', 0x0302),
    D2(0x00FC, 'u', 0x0308),
    D2(0x00FD, 'y', 0x0301),
    D2(0x00FF, 'y', 0x0308),
    /* Latin Extended-A */
    D2(0x0100, 'A', 0x0304),
    D2(0x0101, 'a', 0x0304),
    D2(0x0102, 'A', 0x0306),
    D2(0x0103, 'a', 0x0306),
    D2(0x0104, 'A', 0x0328),
    D2(0x0105, 'a', 0x0328),
    D2(0x0106, 'C', 0x0301),
    D2(0x0107, 'c', 0x0301),
    D2(0x0108, 'C', 0x0302),
    D2(0x0109, 'c', 0x0302),
    D2(0x010A, 'C', 0x0307),
    D2(0x010B, 'c', 0x0307),
    D2(0x010C, 'C', 0x030C),
    D2(0x010D, 'c', 0x030C),
    D2(0x010E, 'D', 0x030C),
    D2(0x010F, 'd', 0x030C),
    D2(0x0112, 'E', 0x0304),
    D2(0x0113, 'e', 0x0304),
    D2(0x0114, 'E', 0x0306),
    D2(0x0115, 'e', 0x0306),
    D2(0x0116, 'E', 0x0307),
    D2(0x0117, 'e', 0x0307),
    D2(0x0118, 'E', 0x0328),
    D2(0x0119, 'e', 0x0328),
    D2(0x011A, 'E', 0x030C),
    D2(0x011B, 'e', 0x030C),
    D2(0x011C, 'G', 0x0302),
    D2(0x011D, 'g', 0x0302),
    D2(0x011E, 'G', 0x0306),
    D2(0x011F, 'g', 0x0306),
    D2(0x0120, 'G', 0x0307),
    D2(0x0121, 'g', 0x0307),
    D2(0x0122, 'G', 0x0327),
    D2(0x0123, 'g', 0x0327),
    D2(0x0124, 'H', 0x0302),
    D2(0x0125, 'h', 0x0302),
    D2(0x0128, 'I', 0x0303),
    D2(0x0129, 'i', 0x0303),
    D2(0x012A, 'I', 0x0304),
    D2(0x012B, 'i', 0x0304),
    D2(0x012C, 'I', 0x0306),
    D2(0x012D, 'i', 0x0306),
    D2(0x012E, 'I', 0x0328),
    D2(0x012F, 'i', 0x0328),
    D2(0x0130, 'I', 0x0307),
    D2(0x0132, 'I', 'J'),
    D2(0x0133, 'i', 'j'),
    D2(0x0134, 'J', 0x0302),
    D2(0x0135, 'j', 0x0302),
    D2(0x0136, 'K', 0x0327),
    D2(0x0137, 'k', 0x0327),
    D2(0x0139, 'L', 0x0301),
    D2(0x013A, 'l', 0x0301),
    D2(0x013B, 'L', 0x0327),
    D2(0x013C, 'l', 0x0327),
    D2(0x013D, 'L', 0x030C),
    D2(0x013E, 'l', 0x030C),
    D2(0x013F, 'L', 0x00B7),
    D2(0x0140, 'l', 0x00B7),
    D2(0x0143, 'N', 0x0301),
    D2(0x0144, 'n', 0x0301),
    D2(0x0145, 'N', 0x0327),
    D2(0x0146, 'n', 0x0327),
    D2(0x0147, 'N', 0x030C),
    D2(0x0148, 'n', 0x030C),
    D2(0x0149, 0x02BC, 'n'),
    D2(0x014C, 'O', 0x0304),
    D2(0x014D, 'o', 0x0304),
    D2(0x014E, 'O', 0x0306),
    D2(0x014F, 'o', 0x0306),
    D2(0x0150, 'O', 0x030B),
    D2(0x0151, 'o', 0x030B),
    D2(0x0154, 'R', 0x0301),
    D2(0x0155, 'r', 0x0301),
    D2(0x0156, 'R', 0x0327),
    D2(0x0157, 'r', 0x0327),
    D2(0x0158, 'R', 0x030C),
    D2(0x0159, 'r', 0x030C),
    D2(0x015A, 'S', 0x0301),
    D2(0x015B, 's', 0x0301),
    D2(0x015C, 'S', 0x0302),
    D2(0x015D, 's', 0x0302),
    D2(0x015E, 'S', 0x0327),
    D2(0x015F, 's', 0x0327),
    D2(0x0160, 'S', 0x030C),
    D2(0x0161, 's', 0x030C),
    D2(0x0162, 'T', 0x0327),
    D2(0x0163, 't', 0x0327),
    D2(0x0164, 'T', 0x030C),
    D2(0x0165, 't', 0x030C),
    D2(0x0168, 'U', 0x0303),
    D2(0x0169, 'u', 0x0303),
    D2(0x016A, 'U', 0x0304),
    D2(0x016B, 'u', 0x0304),
    D2(0x016C, 'U', 0x0306),
    D2(0x016D, 'u', 0x0306),
    D2(0x016E, 'U', 0x030A),
    D2(0x016F, 'u', 0x030A),
    D2(0x0170, 'U', 0x030B),
    D2(0x0171, 'u', 0x030B),
    D2(0x0172, 'U', 0x0328),
    D2(0x0173, 'u', 0x0328),
    D2(0x0174, 'W', 0x0302),
    D2(0x0175, 'w', 0x0302),
    D2(0x0176, 'Y', 0x0302),
    D2(0x0177, 'y', 0x0302),
    D2(0x0178, 'Y', 0x0308),
    D2(0x0179, 'Z', 0x0301),
    D2(0x017A, 'z', 0x0301),
    D2(0x017B, 'Z', 0x0307),
    D2(0x017C, 'z', 0x0307),
    D2(0x017D, 'Z', 0x030C),
    D2(0x017E, 'z', 0x030C),
    D1(0x017F, 's'),
    /* Latin Extended-B */
    D2(0x01A0, 'O', 0x031B),
    D2(0x01A1, 'o', 0x031B),
    D2(0x01AF, 'U', 0x031B),
    D2(0x01B0, 'u', 0x031B),
    /* Latin Extended-B digraphs */
    D2(0x01C4, 'D', 0x017D),
    D2(0x01C5, 'D', 0x017E),
    D2(0x01C6, 'd', 0x017E),
    D2(0x01C7, 'L', 'J'),
    D2(0x01C8, 'L', 'j'),
    D2(0x01C9, 'l', 'j'),
    D2(0x01CA, 'N', 'J'),
    D2(0x01CB, 'N', 'j'),
    D2(0x01CC, 'n', 'j'),
    D2(0x01CD, 'A', 0x030C),
    D2(0x01CE, 'a', 0x030C),
    D2(0x01CF, 'I', 0x030C),
    D2(0x01D0, 'i', 0x030C),
    D2(0x01D1, 'O', 0x030C),
    D2(0x01D2, 'o', 0x030C),
    D2(0x01D3, 'U', 0x030C),
    D2(0x01D4, 'u', 0x030C),
    D3(0x01D5, 'U', 0x0308, 0x0304),
    D3(0x01D6, 'u', 0x0308, 0x0304),
    D3(0x01D7, 'U', 0x0308, 0x0301),
    D3(0x01D8, 'u', 0x0308, 0x0301),
    D3(0x01D9, 'U', 0x0308, 0x030C),
    D3(0x01DA, 'u', 0x0308, 0x030C),
    D3(0x01DB, 'U', 0x0308, 0x0300),
    D3(0x01DC, 'u', 0x0308, 0x0300),
    D3(0x01DE, 'A', 0x0308, 0x0304),
    D3(0x01DF, 'a', 0x0308, 0x0304),
    D3(0x01E0, 'A', 0x0307, 0x0304),
    D3(0x01E1, 'a', 0x0307, 0x0304),
    D2(0x01E6, 'G', 0x030C),
    D2(0x01E7, 'g', 0x030C),
    D2(0x01E8, 'K', 0x030C),
    D2(0x01E9, 'k', 0x030C),
    D2(0x01EA, 'O', 0x0328),
    D2(0x01EB, 'o', 0x0328),
    D3(0x01EC, 'O', 0x0328, 0x0304),
    D3(0x01ED, 'o', 0x0328, 0x0304),
    D2(0x01F0, 'j', 0x030C),
    D2(0x01F1, 'D', 'Z'),
    D2(0x01F2, 'D', 'z'),
    D2(0x01F3, 'd', 'z'),
    D2(0x01F4, 'G', 0x0301),
    D2(0x01F5, 'g', 0x0301),
    D2(0x01F8, 'N', 0x0300),
    D2(0x01F9, 'n', 0x0300),
    D3(0x01FA, 'A', 0x030A, 0x0301),
    D3(0x01FB, 'a', 0x030A, 0x0301),
    D2(0x0200, 'A', 0x030F),
    D2(0x0201, 'a', 0x030F),
    D2(0x0202, 'A', 0x0311),
    D2(0x0203, 'a', 0x0311),
    D2(0x0204, 'E', 0x030F),
    D2(0x0205, 'e', 0x030F),
    D2(0x0206, 'E', 0x0311),
    D2(0x0207, 'e', 0x0311),
    D2(0x0208, 'I', 0x030F),
    D2(0x0209, 'i', 0x030F),
    D2(0x020A, 'I', 0x0311),
    D2(0x020B, 'i', 0x0311),
    D2(0x020C, 'O', 0x030F),
    D2(0x020D, 'o', 0x030F),
    D2(0x020E, 'O', 0x0311),
    D2(0x020F, 'o', 0x0311),
    D2(0x0210, 'R', 0x030F),
    D2(0x0211, 'r', 0x030F),
    D2(0x0212, 'R', 0x0311),
    D2(0x0213, 'r', 0x0311),
    D2(0x0214, 'U', 0x030F),
    D2(0x0215, 'u', 0x030F),
    D2(0x0216, 'U', 0x0311),
    D2(0x0217, 'u', 0x0311),
    D2(0x0218, 'S', 0x0326),
    D2(0x0219, 's', 0x0326),
    D2(0x021A, 'T', 0x0326),
    D2(0x021B, 't', 0x0326),
    D2(0x021E, 'H', 0x030C),
    D2(0x021F, 'h', 0x030C),
    D2(0x0226, 'A', 0x0307),
    D2(0x0227, 'a', 0x0307),
    D2(0x0228, 'E', 0x0327),
    D2(0x0229, 'e', 0x0327),
    D3(0x022A, 'O', 0x0308, 0x0304),
    D3(0x022B, 'o', 0x0308, 0x0304),
    D3(0x022C, 'O', 0x0303, 0x0304),
    D3(0x022D, 'o', 0x0303, 0x0304),
    D2(0x022E, 'O', 0x0307),
    D2(0x022F, 'o', 0x0307),
    D3(0x0230, 'O', 0x0307, 0x0304),
    D3(0x0231, 'o', 0x0307, 0x0304),
    D2(0x0232, 'Y', 0x0304),
    D2(0x0233, 'y', 0x0304),
    /* Modifier letters that yield ASCII */
    D1(0x02B0, 'h'),
    D1(0x02B2, 'j'),
    D1(0x02B3, 'r'),
    D1(0x02B7, 'w'),
    D1(0x02B8, 'y'),
    D1(0x02E1, 'l'),
    D1(0x02E2, 's'),
    D1(0x02E3, 'x'),
    /* Phonetic small capitals that yield ASCII */
    D1(0x1D2C, 'A'),
    D1(0x1D2D, 0x00C6),
    D1(0x1D2E, 'B'),
    D1(0x1D30, 'D'),
    D1(0x1D31, 'E'),
    D1(0x1D33, 'G'),
    D1(0x1D34, 'H'),
    D1(0x1D35, 'I'),
    D1(0x1D36, 'J'),
    D1(0x1D37, 'K'),
    D1(0x1D38, 'L'),
    D1(0x1D39, 'M'),
    D1(0x1D3A, 'N'),
    D1(0x1D3C, 'O'),
    D1(0x1D3E, 'P'),
    D1(0x1D3F, 'R'),
    D1(0x1D40, 'T'),
    D1(0x1D41, 'U'),
    D1(0x1D42, 'W'),
    D1(0x1D43, 'a'),
    D1(0x1D47, 'b'),
    D1(0x1D48, 'd'),
    D1(0x1D49, 'e'),
    D1(0x1D4D, 'g'),
    D1(0x1D4F, 'k'),
    D1(0x1D50, 'm'),
    D1(0x1D52, 'o'),
    D1(0x1D56, 'p'),
    D1(0x1D57, 't'),
    D1(0x1D58, 'u'),
    D1(0x1D5B, 'v'),
    D1(0x1D62, 'i'),
    D1(0x1D63, 'r'),
    D1(0x1D64, 'u'),
    D1(0x1D65, 'v'),
    D1(0x1D9C, 'c'),
    D1(0x1DA0, 'f'),
    D1(0x1DBB, 'z'),
    /* Latin Extended Additional */
    D2(0x1E00, 'A', 0x0325),
    D2(0x1E01, 'a', 0x0325),
    D2(0x1E02, 'B', 0x0307),
    D2(0x1E03, 'b', 0x0307),
    D2(0x1E04, 'B', 0x0323),
    D2(0x1E05, 'b', 0x0323),
    D2(0x1E06, 'B', 0x0331),
    D2(0x1E07, 'b', 0x0331),
    D3(0x1E08, 'C', 0x0327, 0x0301),
    D3(0x1E09, 'c', 0x0327, 0x0301),
    D2(0x1E0A, 'D', 0x0307),
    D2(0x1E0B, 'd', 0x0307),
    D2(0x1E0C, 'D', 0x0323),
    D2(0x1E0D, 'd', 0x0323),
    D2(0x1E0E, 'D', 0x0331),
    D2(0x1E0F, 'd', 0x0331),
    D2(0x1E10, 'D', 0x0327),
    D2(0x1E11, 'd', 0x0327),
    D2(0x1E12, 'D', 0x032D),
    D2(0x1E13, 'd', 0x032D),
    D3(0x1E14, 'E', 0x0304, 0x0300),
    D3(0x1E15, 'e', 0x0304, 0x0300),
    D3(0x1E16, 'E', 0x0304, 0x0301),
    D3(0x1E17, 'e', 0x0304, 0x0301),
    D2(0x1E18, 'E', 0x032D),
    D2(0x1E19, 'e', 0x032D),
    D2(0x1E1A, 'E', 0x0330),
    D2(0x1E1B, 'e', 0x0330),
    D3(0x1E1C, 'E', 0x0327, 0x0306),
    D3(0x1E1D, 'e', 0x0327, 0x0306),
    D2(0x1E1E, 'F', 0x0307),
    D2(0x1E1F, 'f', 0x0307),
    D2(0x1E20, 'G', 0x0304),
    D2(0x1E21, 'g', 0x0304),
    D2(0x1E22, 'H', 0x0307),
    D2(0x1E23, 'h', 0x0307),
    D2(0x1E24, 'H', 0x0323),
    D2(0x1E25, 'h', 0x0323),
    D2(0x1E26, 'H', 0x0308),
    D2(0x1E27, 'h', 0x0308),
    D2(0x1E28, 'H', 0x0327),
    D2(0x1E29, 'h', 0x0327),
    D2(0x1E2A, 'H', 0x032E),
    D2(0x1E2B, 'h', 0x032E),
    D2(0x1E2C, 'I', 0x0330),
    D2(0x1E2D, 'i', 0x0330),
    D3(0x1E2E, 'I', 0x0308, 0x0301),
    D3(0x1E2F, 'i', 0x0308, 0x0301),
    D2(0x1E30, 'K', 0x0301),
    D2(0x1E31, 'k', 0x0301),
    D2(0x1E32, 'K', 0x0323),
    D2(0x1E33, 'k', 0x0323),
    D2(0x1E34, 'K', 0x0331),
    D2(0x1E35, 'k', 0x0331),
    D2(0x1E36, 'L', 0x0323),
    D2(0x1E37, 'l', 0x0323),
    D3(0x1E38, 'L', 0x0323, 0x0304),
    D3(0x1E39, 'l', 0x0323, 0x0304),
    D2(0x1E3A, 'L', 0x0331),
    D2(0x1E3B, 'l', 0x0331),
    D2(0x1E3C, 'L', 0x032D),
    D2(0x1E3D, 'l', 0x032D),
    D2(0x1E3E, 'M', 0x0301),
    D2(0x1E3F, 'm', 0x0301),
    D2(0x1E40, 'M', 0x0307),
    D2(0x1E41, 'm', 0x0307),
    D2(0x1E42, 'M', 0x0323),
    D2(0x1E43, 'm', 0x0323),
    D2(0x1E44, 'N', 0x0307),
    D2(0x1E45, 'n', 0x0307),
    D2(0x1E46, 'N', 0x0323),
    D2(0x1E47, 'n', 0x0323),
    D2(0x1E48, 'N', 0x0331),
    D2(0x1E49, 'n', 0x0331),
    D2(0x1E4A, 'N', 0x032D),
    D2(0x1E4B, 'n', 0x032D),
    D3(0x1E4C, 'O', 0x0303, 0x0301),
    D3(0x1E4D, 'o', 0x0303, 0x0301),
    D3(0x1E4E, 'O', 0x0303, 0x0308),
    D3(0x1E4F, 'o', 0x0303, 0x0308),
    D3(0x1E50, 'O', 0x0304, 0x0300),
    D3(0x1E51, 'o', 0x0304, 0x0300),
    D3(0x1E52, 'O', 0x0304, 0x0301),
    D3(0x1E53, 'o', 0x0304, 0x0301),
    D2(0x1E54, 'P', 0x0301),
    D2(0x1E55, 'p', 0x0301),
    D2(0x1E56, 'P', 0x0307),
    D2(0x1E57, 'p', 0x0307),
    D2(0x1E58, 'R', 0x0307),
    D2(0x1E59, 'r', 0x0307),
    D2(0x1E5A, 'R', 0x0323),
    D2(0x1E5B, 'r', 0x0323),
    D3(0x1E5C, 'R', 0x0323, 0x0304),
    D3(0x1E5D, 'r', 0x0323, 0x0304),
    D2(0x1E5E, 'R', 0x0331),
    D2(0x1E5F, 'r', 0x0331),
    D2(0x1E60, 'S', 0x0307),
    D2(0x1E61, 's', 0x0307),
    D2(0x1E62, 'S', 0x0323),
    D2(0x1E63, 's', 0x0323),
    D3(0x1E64, 'S', 0x0301, 0x0307),
    D3(0x1E65, 's', 0x0301, 0x0307),
    D3(0x1E66, 'S', 0x030C, 0x0307),
    D3(0x1E67, 's', 0x030C, 0x0307),
    D3(0x1E68, 'S', 0x0323, 0x0307),
    D3(0x1E69, 's', 0x0323, 0x0307),
    D2(0x1E6A, 'T', 0x0307),
    D2(0x1E6B, 't', 0x0307),
    D2(0x1E6C, 'T', 0x0323),
    D2(0x1E6D, 't', 0x0323),
    D2(0x1E6E, 'T', 0x0331),
    D2(0x1E6F, 't', 0x0331),
    D2(0x1E70, 'T', 0x032D),
    D2(0x1E71, 't', 0x032D),
    D2(0x1E72, 'U', 0x0324),
    D2(0x1E73, 'u', 0x0324),
    D2(0x1E74, 'U', 0x0330),
    D2(0x1E75, 'u', 0x0330),
    D2(0x1E76, 'U', 0x032D),
    D2(0x1E77, 'u', 0x032D),
    D3(0x1E78, 'U', 0x0303, 0x0301),
    D3(0x1E79, 'u', 0x0303, 0x0301),
    D3(0x1E7A, 'U', 0x0304, 0x0308),
    D3(0x1E7B, 'u', 0x0304, 0x0308),
    D2(0x1E7C, 'V', 0x0303),
    D2(0x1E7D, 'v', 0x0303),
    D2(0x1E7E, 'V', 0x0323),
    D2(0x1E7F, 'v', 0x0323),
    D2(0x1E80, 'W', 0x0300),
    D2(0x1E81, 'w', 0x0300),
    D2(0x1E82, 'W', 0x0301),
    D2(0x1E83, 'w', 0x0301),
    D2(0x1E84, 'W', 0x0308),
    D2(0x1E85, 'w', 0x0308),
    D2(0x1E86, 'W', 0x0307),
    D2(0x1E87, 'w', 0x0307),
    D2(0x1E88, 'W', 0x0323),
    D2(0x1E89, 'w', 0x0323),
    D2(0x1E8A, 'X', 0x0307),
    D2(0x1E8B, 'x', 0x0307),
    D2(0x1E8C, 'X', 0x0308),
    D2(0x1E8D, 'x', 0x0308),
    D2(0x1E8E, 'Y', 0x0307),
    D2(0x1E8F, 'y', 0x0307),
    D2(0x1E90, 'Z', 0x0302),
    D2(0x1E91, 'z', 0x0302),
    D2(0x1E92, 'Z', 0x0323),
    D2(0x1E93, 'z', 0x0323),
    D2(0x1E94, 'Z', 0x0331),
    D2(0x1E95, 'z', 0x0331),
    D2(0x1E96, 'h', 0x0331),
    D2(0x1E97, 't', 0x0308),
    D2(0x1E98, 'w', 0x030A),
    D2(0x1E99, 'y', 0x030A),
    D2(0x1E9A, 'a', 0x02BE),
    D2(0x1E9B, 's', 0x0307),
    D2(0x1EA0, 'A', 0x0323),
    D2(0x1EA1, 'a', 0x0323),
    D2(0x1EA2, 'A', 0x0309),
    D2(0x1EA3, 'a', 0x0309),
    D3(0x1EA4, 'A', 0x0302, 0x0301),
    D3(0x1EA5, 'a', 0x0302, 0x0301),
    D3(0x1EA6, 'A', 0x0302, 0x0300),
    D3(0x1EA7, 'a', 0x0302, 0x0300),
    D3(0x1EA8, 'A', 0x0302, 0x0309),
    D3(0x1EA9, 'a', 0x0302, 0x0309),
    D3(0x1EAA, 'A', 0x0302, 0x0303),
    D3(0x1EAB, 'a', 0x0302, 0x0303),
    D3(0x1EAC, 'A', 0x0323, 0x0302),
    D3(0x1EAD, 'a', 0x0323, 0x0302),
    D3(0x1EAE, 'A', 0x0306, 0x0301),
    D3(0x1EAF, 'a', 0x0306, 0x0301),
    D3(0x1EB0, 'A', 0x0306, 0x0300),
    D3(0x1EB1, 'a', 0x0306, 0x0300),
    D3(0x1EB2, 'A', 0x0306, 0x0309),
    D3(0x1EB3, 'a', 0x0306, 0x0309),
    D3(0x1EB4, 'A', 0x0306, 0x0303),
    D3(0x1EB5, 'a', 0x0306, 0x0303),
    D3(0x1EB6, 'A', 0x0323, 0x0306),
    D3(0x1EB7, 'a', 0x0323, 0x0306),
    D2(0x1EB8, 'E', 0x0323),
    D2(0x1EB9, 'e', 0x0323),
    D2(0x1EBA, 'E', 0x0309),
    D2(0x1EBB, 'e', 0x0309),
    D2(0x1EBC, 'E', 0x0303),
    D2(0x1EBD, 'e', 0x0303),
    D3(0x1EBE, 'E', 0x0302, 0x0301),
    D3(0x1EBF, 'e', 0x0302, 0x0301),
    D3(0x1EC0, 'E', 0x0302, 0x0300),
    D3(0x1EC1, 'e', 0x0302, 0x0300),
    D3(0x1EC2, 'E', 0x0302, 0x0309),
    D3(0x1EC3, 'e', 0x0302, 0x0309),
    D3(0x1EC4, 'E', 0x0302, 0x0303),
    D3(0x1EC5, 'e', 0x0302, 0x0303),
    D3(0x1EC6, 'E', 0x0323, 0x0302),
    D3(0x1EC7, 'e', 0x0323, 0x0302),
    D2(0x1EC8, 'I', 0x0309),
    D2(0x1EC9, 'i', 0x0309),
    D2(0x1ECA, 'I', 0x0323),
    D2(0x1ECB, 'i', 0x0323),
    D2(0x1ECC, 'O', 0x0323),
    D2(0x1ECD, 'o', 0x0323),
    D2(0x1ECE, 'O', 0x0309),
    D2(0x1ECF, 'o', 0x0309),
    D3(0x1ED0, 'O', 0x0302, 0x0301),
    D3(0x1ED1, 'o', 0x0302, 0x0301),
    D3(0x1ED2, 'O', 0x0302, 0x0300),
    D3(0x1ED3, 'o', 0x0302, 0x0300),
    D3(0x1ED4, 'O', 0x0302, 0x0309),
    D3(0x1ED5, 'o', 0x0302, 0x0309),
    D3(0x1ED6, 'O', 0x0302, 0x0303),
    D3(0x1ED7, 'o', 0x0302, 0x0303),
    D3(0x1ED8, 'O', 0x0323, 0x0302),
    D3(0x1ED9, 'o', 0x0323, 0x0302),
    D3(0x1EDA, 'O', 0x031B, 0x0301),
    D3(0x1EDB, 'o', 0x031B, 0x0301),
    D3(0x1EDC, 'O', 0x031B, 0x0300),
    D3(0x1EDD, 'o', 0x031B, 0x0300),
    D3(0x1EDE, 'O', 0x031B, 0x0309),
    D3(0x1EDF, 'o', 0x031B, 0x0309),
    D3(0x1EE0, 'O', 0x031B, 0x0303),
    D3(0x1EE1, 'o', 0x031B, 0x0303),
    D3(0x1EE2, 'O', 0x031B, 0x0323),
    D3(0x1EE3, 'o', 0x031B, 0x0323),
    D2(0x1EE4, 'U', 0x0323),
    D2(0x1EE5, 'u', 0x0323),
    D2(0x1EE6, 'U', 0x0309),
    D2(0x1EE7, 'u', 0x0309),
    D3(0x1EE8, 'U', 0x031B, 0x0301),
    D3(0x1EE9, 'u', 0x031B, 0x0301),
    D3(0x1EEA, 'U', 0x031B, 0x0300),
    D3(0x1EEB, 'u', 0x031B, 0x0300),
    D3(0x1EEC, 'U', 0x031B, 0x0309),
    D3(0x1EED, 'u', 0x031B, 0x0309),
    D3(0x1EEE, 'U', 0x031B, 0x0303),
    D3(0x1EEF, 'u', 0x031B, 0x0303),
    D3(0x1EF0, 'U', 0x031B, 0x0323),
    D3(0x1EF1, 'u', 0x031B, 0x0323),
    D2(0x1EF2, 'Y', 0x0300),
    D2(0x1EF3, 'y', 0x0300),
    D2(0x1EF4, 'Y', 0x0323),
    D2(0x1EF5, 'y', 0x0323),
    D2(0x1EF6, 'Y', 0x0309),
    D2(0x1EF7, 'y', 0x0309),
    D2(0x1EF8, 'Y', 0x0303),
    D2(0x1EF9, 'y', 0x0303),
    /* Superscripts and subscripts */
    D1(0x2070, '0'),
    D1(0x2071, 'i'),
    D1(0x2074, '4'),
    D1(0x2075, '5'),
    D1(0x2076, '6'),
    D1(0x2077, '7'),
    D1(0x2078, '8'),
    D1(0x2079, '9'),
    D1(0x207A, '+'),
    D1(0x207B, '-'),
    D1(0x207C, '='),
    D1(0x207D, '('),
    D1(0x207E, ')'),
    D1(0x207F, 'n'),
    D1(0x2080, '0'),
    D1(0x2081, '1'),
    D1(0x2082, '2'),
    D1(0x2083, '3'),
    D1(0x2084, '4'),
    D1(0x2085, '5'),
    D1(0x2086, '6'),
    D1(0x2087, '7'),
    D1(0x2088, '8'),
    D1(0x2089, '9'),
    D1(0x208A, '+'),
    D1(0x208B, '-'),
    D1(0x208C, '='),
    D1(0x208D, '('),
    D1(0x208E, ')'),
    D1(0x2090, 'a'),
    D1(0x2091, 'e'),
    D1(0x2092, 'o'),
    D1(0x2093, 'x'),
    D1(0x2095, 'h'),
    D1(0x2096, 'k'),
    D1(0x2097, 'l'),
    D1(0x2098, 'm'),
    D1(0x2099, 'n'),
    D1(0x209A, 'p'),
    D1(0x209B, 's'),
    D1(0x209C, 't'),
    D2(0x20A8, 'R', 's'),
    /* Letterlike symbols */
    D3(0x2100, 'a', '/', 'c'),
    D3(0x2101, 'a', '/', 's'),
    D1(0x2102, 'C'),
    D2(0x2103, 0x00B0, 'C'),
    D3(0x2105, 'c', '/', 'o'),
    D3(0x2106, 'c', '/', 'u'),
    D2(0x2109, 0x00B0, 'F'),
    D1(0x210A, 'g'),
    D1(0x210B, 'H'),
    D1(0x210C, 'H'),
    D1(0x210D, 'H'),
    D1(0x210E, 'h'),
    D1(0x2110, 'I'),
    D1(0x2111, 'I'),
    D1(0x2112, 'L'),
    D1(0x2113, 'l'),
    D1(0x2115, 'N'),
    D2(0x2116, 'N', 'o'),
    D1(0x2119, 'P'),
    D1(0x211A, 'Q'),
    D1(0x211B, 'R'),
    D1(0x211C, 'R'),
    D1(0x211D, 'R'),
    D2(0x2120, 'S', 'M'),
    D3(0x2121, 'T', 'E', 'L'),
    D2(0x2122, 'T', 'M'),
    D1(0x2124, 'Z'),
    D1(0x2128, 'Z'),
    D1(0x212A, 'K'),
    D2(0x212B, 'A', 0x030A),
    D1(0x212C, 'B'),
    D1(0x212D, 'C'),
    D1(0x212F, 'e'),
    D1(0x2130, 'E'),
    D1(0x2131, 'F'),
    D1(0x2133, 'M'),
    D1(0x2134, 'o'),
    D1(0x2139, 'i'),
    D3(0x213B, 'F', 'A', 'X'),
    D1(0x2145, 'D'),
    D1(0x2146, 'd'),
    D1(0x2147, 'e'),
    D1(0x2148, 'i'),
    D1(0x2149, 'j'),
    /* Vulgar fractions */
    D3(0x2150, '1', 0x2044, '7'),
    D3(0x2151, '1', 0x2044, '9'),
    D4(0x2152, '1', 0x2044, '1', '0'),
    D3(0x2153, '1', 0x2044, '3'),
    D3(0x2154, '2', 0x2044, '3'),
    D3(0x2155, '1', 0x2044, '5'),
    D3(0x2156, '2', 0x2044, '5'),
    D3(0x2157, '3', 0x2044, '5'),
    D3(0x2158, '4', 0x2044, '5'),
    D3(0x2159, '1', 0x2044, '6'),
    D3(0x215A, '5', 0x2044, '6'),
    D3(0x215B, '1', 0x2044, '8'),
    D3(0x215C, '3', 0x2044, '8'),
    D3(0x215D, '5', 0x2044, '8'),
    D3(0x215E, '7', 0x2044, '8'),
    D2(0x215F, '1', 0x2044),
    /* Roman numerals */
    D1(0x2160, 'I'),
    D2(0x2161, 'I', 'I'),
    D3(0x2162, 'I', 'I', 'I'),
    D2(0x2163, 'I', 'V'),
    D1(0x2164, 'V'),
    D2(0x2165, 'V', 'I'),
    D3(0x2166, 'V', 'I', 'I'),
    D4(0x2167, 'V', 'I', 'I', 'I'),
    D2(0x2168, 'I', 'X'),
    D1(0x2169, 'X'),
    D2(0x216A, 'X', 'I'),
    D3(0x216B, 'X', 'I', 'I'),
    D1(0x216C, 'L'),
    D1(0x216D, 'C'),
    D1(0x216E, 'D'),
    D1(0x216F, 'M'),
    D1(0x2170, 'i'),
    D2(0x2171, 'i', 'i'),
    D3(0x2172, 'i', 'i', 'i'),
    D2(0x2173, 'i', 'v'),
    D1(0x2174, 'v'),
    D2(0x2175, 'v', 'i'),
    D3(0x2176, 'v', 'i', 'i'),
    D4(0x2177, 'v', 'i', 'i', 'i'),
    D2(0x2178, 'i', 'x'),
    D1(0x2179, 'x'),
    D2(0x217A, 'x', 'i'),
    D3(0x217B, 'x', 'i', 'i'),
    D1(0x217C, 'l'),
    D1(0x217D, 'c'),
    D1(0x217E, 'd'),
    D1(0x217F, 'm'),
    D3(0x2189, '0', 0x2044, '3'),
    /* Latin Extended-C */
    D1(0x2C7C, 'j'),
    D1(0x2C7D, 'V'),
    /* Enclosed CJK letters and months */
    D3(0x3250, 'P', 'T', 'E'),
    D2(0x3251, '2', '1'),
    D2(0x3252, '2', '2'),
    D2(0x3253, '2', '3'),
    D2(0x3254, '2', '4'),
    D2(0x3255, '2', '5'),
    D2(0x3256, '2', '6'),
    D2(0x3257, '2', '7'),
    D2(0x3258, '2', '8'),
    D2(0x3259, '2', '9'),
    D2(0x325A, '3', '0'),
    D2(0x325B, '3', '1'),
    D2(0x325C, '3', '2'),
    D2(0x325D, '3', '3'),
    D2(0x325E, '3', '4'),
    D2(0x325F, '3', '5'),
    D2(0x32B1, '3', '6'),
    D2(0x32B2, '3', '7'),
    D2(0x32B3, '3', '8'),
    D2(0x32B4, '3', '9'),
    D2(0x32B5, '4', '0'),
    D2(0x32B6, '4', '1'),
    D2(0x32B7, '4', '2'),
    D2(0x32B8, '4', '3'),
    D2(0x32B9, '4', '4'),
    D2(0x32BA, '4', '5'),
    D2(0x32BB, '4', '6'),
    D2(0x32BC, '4', '7'),
    D2(0x32BD, '4', '8'),
    D2(0x32BE, '4', '9'),
    D2(0x32BF, '5', '0'),
    D2(0x32C0, '1', 0x6708),
    D2(0x32C1, '2', 0x6708),
    D2(0x32C2, '3', 0x6708),
    D2(0x32C3, '4', 0x6708),
    D2(0x32C4, '5', 0x6708),
    D2(0x32C5, '6', 0x6708),
    D2(0x32C6, '7', 0x6708),
    D2(0x32C7, '8', 0x6708),
    D2(0x32C8, '9', 0x6708),
    D3(0x32C9, '1', '0', 0x6708),
    D3(0x32CA, '1', '1', 0x6708),
    D3(0x32CB, '1', '2', 0x6708),
    D2(0x32CC, 'H', 'g'),
    D3(0x32CD, 'e', 'r', 'g'),
    D2(0x32CE, 'e', 'V'),
    D3(0x32CF, 'L', 'T', 'D'),
    /* CJK Compatibility: squared units */
    D3(0x3371, 'h', 'P', 'a'),
    D2(0x3372, 'd', 'a'),
    D2(0x3373, 'A', 'U'),
    D3(0x3374, 'b', 'a', 'r'),
    D2(0x3375, 'o', 'V'),
    D2(0x3376, 'p', 'c'),
    D2(0x3377, 'd', 'm'),
    D3(0x3378, 'd', 'm', '2'),
    D3(0x3379, 'd', 'm', '3'),
    D2(0x337A, 'I', 'U'),
    D2(0x3380, 'p', 'A'),
    D2(0x3381, 'n', 'A'),
    D2(0x3382, 0x03BC, 'A'),
    D2(0x3383, 'm', 'A'),
    D2(0x3384, 'k', 'A'),
    D2(0x3385, 'K', 'B'),
    D2(0x3386, 'M', 'B'),
    D2(0x3387, 'G', 'B'),
    D3(0x3388, 'c', 'a', 'l'),
    D4(0x3389, 'k', 'c', 'a', 'l'),
    D2(0x338A, 'p', 'F'),
    D2(0x338B, 'n', 'F'),
    D2(0x338C, 0x03BC, 'F'),
    D2(0x338D, 0x03BC, 'g'),
    D2(0x338E, 'm', 'g'),
    D2(0x338F, 'k', 'g'),
    D2(0x3390, 'H', 'z'),
    D3(0x3391, 'k', 'H', 'z'),
    D3(0x3392, 'M', 'H', 'z'),
    D3(0x3393, 'G', 'H', 'z'),
    D3(0x3394, 'T', 'H', 'z'),
    D2(0x3395, 0x03BC, 'l'),
    D2(0x3396, 'm', 'l'),
    D2(0x3397, 'd', 'l'),
    D2(0x3398, 'k', 'l'),
    D2(0x3399, 'f', 'm'),
    D2(0x339A, 'n', 'm'),
    D2(0x339B, 0x03BC, 'm'),
    D2(0x339C, 'm', 'm'),
    D2(0x339D, 'c', 'm'),
    D2(0x339E, 'k', 'm'),
    D3(0x339F, 'm', 'm', '2'),
    D3(0x33A0, 'c', 'm', '2'),
    D2(0x33A1, 'm', '2'),
    D3(0x33A2, 'k', 'm', '2'),
    D3(0x33A3, 'm', 'm', '3'),
    D3(0x33A4, 'c', 'm', '3'),
    D2(0x33A5, 'm', '3'),
    D3(0x33A6, 'k', 'm', '3'),
    D3(0x33A7, 'm', 0x2215, 's'),
    D4(0x33A8, 'm', 0x2215, 's', '2'),
    D2(0x33A9, 'P', 'a'),
    D3(0x33AA, 'k', 'P', 'a'),
    D3(0x33AB, 'M', 'P', 'a'),
    D3(0x33AC, 'G', 'P', 'a'),
    D3(0x33AD, 'r', 'a', 'd'),
    D2(0x33B0, 'p', 's'),
    D2(0x33B1, 'n', 's'),
    D2(0x33B2, 0x03BC, 's'),
    D2(0x33B3, 'm', 's'),
    D2(0x33B4, 'p', 'V'),
    D2(0x33B5, 'n', 'V'),
    D2(0x33B6, 0x03BC, 'V'),
    D2(0x33B7, 'm', 'V'),
    D2(0x33B8, 'k', 'V'),
    D2(0x33B9, 'M', 'V'),
    D2(0x33BA, 'p', 'W'),
    D2(0x33BB, 'n', 'W'),
    D2(0x33BC, 0x03BC, 'W'),
    D2(0x33BD, 'm', 'W'),
    D2(0x33BE, 'k', 'W'),
    D2(0x33BF, 'M', 'W'),
    D2(0x33C0, 'k', 0x03A9),
    D2(0x33C1, 'M', 0x03A9),
    D4(0x33C2, 'a', '.', 'm', '.'),
    D2(0x33C3, 'B', 'q'),
    D2(0x33C4, 'c', 'c'),
    D2(0x33C5, 'c', 'd'),
    D4(0x33C6, 'C', 0x2215, 'k', 'g'),
    D3(0x33C7, 'C', 'o', '.'),
    D2(0x33C8, 'd', 'B'),
    D2(0x33C9, 'G', 'y'),
    D2(0x33CA, 'h', 'a'),
    D2(0x33CB, 'H', 'P'),
    D2(0x33CC, 'i', 'n'),
    D2(0x33CD, 'K', 'K'),
    D2(0x33CE, 'K', 'M'),
    D2(0x33CF, 'k', 't'),
    D2(0x33D0, 'l', 'm'),
    D2(0x33D1, 'l', 'n'),
    D3(0x33D2, 'l', 'o', 'g'),
    D2(0x33D3, 'l', 'x'),
    D2(0x33D4, 'm', 'b'),
    D3(0x33D5, 'm', 'i', 'l'),
    D3(0x33D6, 'm', 'o', 'l'),
    D2(0x33D7, 'P', 'H'),
    D4(0x33D8, 'p', '.', 'm', '.'),
    D3(0x33D9, 'P', 'P', 'M'),
    D2(0x33DA, 'P', 'R'),
    D2(0x33DB, 's', 'r'),
    D2(0x33DC, 'S', 'v'),
    D2(0x33DD, 'W', 'b'),
    D3(0x33DE, 'V', 0x2215, 'm'),
    D3(0x33DF, 'A', 0x2215, 'm'),
    D3(0x33FF, 'g', 'a', 'l'),
    /* Latin Extended-D */
    D1(0xA7F1, 'S'),
    D1(0xA7F2, 'C'),
    D1(0xA7F3, 'F'),
    D1(0xA7F4, 'Q'),
    /* Ligatures */
    D2(0xFB00, 'f', 'f'),
    D2(0xFB01, 'f', 'i'),
    D2(0xFB02, 'f', 'l'),
    D3(0xFB03, 'f', 'f', 'i'),
    D3(0xFB04, 'f', 'f', 'l'),
    D2(0xFB05, 's', 't'),
    D2(0xFB06, 's', 't'),
    D1(0xFF9E, 0x3099),
    D1(0xFF9F, 0x309A),
    D1(0x107A5, 'q'),
    /* Enclosed Alphanumeric Supplement */
    D3(0x1F12A, 0x3014, 'S', 0x3015),
    D1(0x1F12B, 'C'),
    D1(0x1F12C, 'R'),
    D2(0x1F12D, 'C', 'D'),
    D2(0x1F12E, 'W', 'Z'),
    D2(0x1F14A, 'H', 'V'),
    D2(0x1F14B, 'M', 'V'),
    D2(0x1F14C, 'S', 'D'),
    D2(0x1F14D, 'S', 'S'),
    D3(0x1F14E, 'P', 'P', 'V'),
    D2(0x1F14F, 'W', 'C'),
    D2(0x1F16A, 'M', 'C'),
    D2(0x1F16B, 'M', 'D'),
    D2(0x1F16C, 'M', 'R'),
    D2(0x1F190, 'D', 'J'),
};

static const Decomp *decomp_find(uint32_t cp) {
  size_t lo = 0, hi = sizeof(DECOMP) / sizeof(DECOMP[0]);
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (DECOMP[mid].cp < cp) lo = mid + 1;
    else if (DECOMP[mid].cp > cp) hi = mid;
    else return &DECOMP[mid];
  }
  return NULL;
}

/* Combining marks (\p{M}): ranges, sorted. */
static const uint32_t MARK_RANGES[][2] = {
    {0x0300, 0x036F}, {0x0483, 0x0489}, {0x0591, 0x05BD}, {0x05BF, 0x05BF}, {0x05C1, 0x05C2},
    {0x05C4, 0x05C5}, {0x05C7, 0x05C7}, {0x0610, 0x061A}, {0x064B, 0x065F}, {0x0670, 0x0670},
    {0x06D6, 0x06DC}, {0x06DF, 0x06E4}, {0x06E7, 0x06E8}, {0x06EA, 0x06ED}, {0x0711, 0x0711},
    {0x0730, 0x074A}, {0x07A6, 0x07B0}, {0x07EB, 0x07F3}, {0x07FD, 0x07FD}, {0x0816, 0x0819},
    {0x081B, 0x0823}, {0x0825, 0x0827}, {0x0829, 0x082D}, {0x0859, 0x085B}, {0x0897, 0x089F},
    {0x08CA, 0x08E1}, {0x08E3, 0x0903}, {0x093A, 0x093C}, {0x093E, 0x094F}, {0x0951, 0x0957},
    {0x0962, 0x0963}, {0x0981, 0x0983}, {0x09BC, 0x09BC}, {0x09BE, 0x09C4}, {0x09C7, 0x09C8},
    {0x09CB, 0x09CD}, {0x09D7, 0x09D7}, {0x09E2, 0x09E3}, {0x09FE, 0x09FE}, {0x0A01, 0x0A03},
    {0x0A3C, 0x0A3C}, {0x0A3E, 0x0A42}, {0x0A47, 0x0A48}, {0x0A4B, 0x0A4D}, {0x0A51, 0x0A51},
    {0x0A70, 0x0A71}, {0x0A75, 0x0A75}, {0x0A81, 0x0A83}, {0x0ABC, 0x0ABC}, {0x0ABE, 0x0AC5},
    {0x0AC7, 0x0AC9}, {0x0ACB, 0x0ACD}, {0x0AE2, 0x0AE3}, {0x0AFA, 0x0AFF}, {0x0B01, 0x0B03},
    {0x0B3C, 0x0B3C}, {0x0B3E, 0x0B44}, {0x0B47, 0x0B48}, {0x0B4B, 0x0B4D}, {0x0B55, 0x0B57},
    {0x0B62, 0x0B63}, {0x0B82, 0x0B82}, {0x0BBE, 0x0BC2}, {0x0BC6, 0x0BC8}, {0x0BCA, 0x0BCD},
    {0x0BD7, 0x0BD7}, {0x0C00, 0x0C04}, {0x0C3C, 0x0C3C}, {0x0C3E, 0x0C44}, {0x0C46, 0x0C48},
    {0x0C4A, 0x0C4D}, {0x0C55, 0x0C56}, {0x0C62, 0x0C63}, {0x0C81, 0x0C83}, {0x0CBC, 0x0CBC},
    {0x0CBE, 0x0CC4}, {0x0CC6, 0x0CC8}, {0x0CCA, 0x0CCD}, {0x0CD5, 0x0CD6}, {0x0CE2, 0x0CE3},
    {0x0CF3, 0x0CF3}, {0x0D00, 0x0D03}, {0x0D3B, 0x0D3C}, {0x0D3E, 0x0D44}, {0x0D46, 0x0D48},
    {0x0D4A, 0x0D4D}, {0x0D57, 0x0D57}, {0x0D62, 0x0D63}, {0x0D81, 0x0D83}, {0x0DCA, 0x0DCA},
    {0x0DCF, 0x0DD4}, {0x0DD6, 0x0DD6}, {0x0DD8, 0x0DDF}, {0x0DF2, 0x0DF3}, {0x0E31, 0x0E31},
    {0x0E34, 0x0E3A}, {0x0E47, 0x0E4E}, {0x0EB1, 0x0EB1}, {0x0EB4, 0x0EBC}, {0x0EC8, 0x0ECE},
    {0x0F18, 0x0F19}, {0x0F35, 0x0F35}, {0x0F37, 0x0F37}, {0x0F39, 0x0F39}, {0x0F3E, 0x0F3F},
    {0x0F71, 0x0F84}, {0x0F86, 0x0F87}, {0x0F8D, 0x0F97}, {0x0F99, 0x0FBC}, {0x0FC6, 0x0FC6},
    {0x102B, 0x103E}, {0x1056, 0x1059}, {0x105E, 0x1060}, {0x1062, 0x1064}, {0x1067, 0x106D},
    {0x1071, 0x1074}, {0x1082, 0x108D}, {0x108F, 0x108F}, {0x109A, 0x109D}, {0x135D, 0x135F},
    {0x1712, 0x1715}, {0x1732, 0x1734}, {0x1752, 0x1753}, {0x1772, 0x1773}, {0x17B4, 0x17D3},
    {0x17DD, 0x17DD}, {0x180B, 0x180D}, {0x180F, 0x180F}, {0x1885, 0x1886}, {0x18A9, 0x18A9},
    {0x1920, 0x192B}, {0x1930, 0x193B}, {0x1A17, 0x1A1B}, {0x1A55, 0x1A5E}, {0x1A60, 0x1A7C},
    {0x1A7F, 0x1A7F}, {0x1AB0, 0x1ADD}, {0x1AE0, 0x1AEB}, {0x1B00, 0x1B04}, {0x1B34, 0x1B44},
    {0x1B6B, 0x1B73}, {0x1B80, 0x1B82}, {0x1BA1, 0x1BAD}, {0x1BE6, 0x1BF3}, {0x1C24, 0x1C37},
    {0x1CD0, 0x1CD2}, {0x1CD4, 0x1CE8}, {0x1CED, 0x1CED}, {0x1CF4, 0x1CF4}, {0x1CF7, 0x1CF9},
    {0x1DC0, 0x1DFF}, {0x20D0, 0x20F0}, {0x2CEF, 0x2CF1}, {0x2D7F, 0x2D7F}, {0x2DE0, 0x2DFF},
    {0x302A, 0x302F}, {0x3099, 0x309A}, {0xA66F, 0xA672}, {0xA674, 0xA67D}, {0xA69E, 0xA69F},
    {0xA6F0, 0xA6F1}, {0xA802, 0xA802}, {0xA806, 0xA806}, {0xA80B, 0xA80B}, {0xA823, 0xA827},
    {0xA82C, 0xA82C}, {0xA880, 0xA881}, {0xA8B4, 0xA8C5}, {0xA8E0, 0xA8F1}, {0xA8FF, 0xA8FF},
    {0xA926, 0xA92D}, {0xA947, 0xA953}, {0xA980, 0xA983}, {0xA9B3, 0xA9C0}, {0xA9E5, 0xA9E5},
    {0xAA29, 0xAA36}, {0xAA43, 0xAA43}, {0xAA4C, 0xAA4D}, {0xAA7B, 0xAA7D}, {0xAAB0, 0xAAB0},
    {0xAAB2, 0xAAB4}, {0xAAB7, 0xAAB8}, {0xAABE, 0xAABF}, {0xAAC1, 0xAAC1}, {0xAAEB, 0xAAEF},
    {0xAAF5, 0xAAF6}, {0xABE3, 0xABEA}, {0xABEC, 0xABED}, {0xFB1E, 0xFB1E}, {0xFE00, 0xFE0F},
    {0xFE20, 0xFE2F}, {0x101FD, 0x101FD}, {0x102E0, 0x102E0}, {0x10376, 0x1037A}, {0x10A01, 0x10A03},
    {0x10A05, 0x10A06}, {0x10A0C, 0x10A0F}, {0x10A38, 0x10A3A}, {0x10A3F, 0x10A3F}, {0x10AE5, 0x10AE6},
    {0x10D24, 0x10D27}, {0x10D69, 0x10D6D}, {0x10EAB, 0x10EAC}, {0x10EFA, 0x10EFF}, {0x10F46, 0x10F50},
    {0x10F82, 0x10F85}, {0x11000, 0x11002}, {0x11038, 0x11046}, {0x11070, 0x11070}, {0x11073, 0x11074},
    {0x1107F, 0x11082}, {0x110B0, 0x110BA}, {0x110C2, 0x110C2}, {0x11100, 0x11102}, {0x11127, 0x11134},
    {0x11145, 0x11146}, {0x11173, 0x11173}, {0x11180, 0x11182}, {0x111B3, 0x111C0}, {0x111C9, 0x111CC},
    {0x111CE, 0x111CF}, {0x1122C, 0x11237}, {0x1123E, 0x1123E}, {0x11241, 0x11241}, {0x112DF, 0x112EA},
    {0x11300, 0x11303}, {0x1133B, 0x1133C}, {0x1133E, 0x11344}, {0x11347, 0x11348}, {0x1134B, 0x1134D},
    {0x11357, 0x11357}, {0x11362, 0x11363}, {0x11366, 0x1136C}, {0x11370, 0x11374}, {0x113B8, 0x113C0},
    {0x113C2, 0x113C2}, {0x113C5, 0x113C5}, {0x113C7, 0x113CA}, {0x113CC, 0x113D0}, {0x113D2, 0x113D2},
    {0x113E1, 0x113E2}, {0x11435, 0x11446}, {0x1145E, 0x1145E}, {0x114B0, 0x114C3}, {0x115AF, 0x115B5},
    {0x115B8, 0x115C0}, {0x115DC, 0x115DD}, {0x11630, 0x11640}, {0x116AB, 0x116B7}, {0x1171D, 0x1172B},
    {0x1182C, 0x1183A}, {0x11930, 0x11935}, {0x11937, 0x11938}, {0x1193B, 0x1193E}, {0x11940, 0x11940},
    {0x11942, 0x11943}, {0x119D1, 0x119D7}, {0x119DA, 0x119E0}, {0x119E4, 0x119E4}, {0x11A01, 0x11A0A},
    {0x11A33, 0x11A39}, {0x11A3B, 0x11A3E}, {0x11A47, 0x11A47}, {0x11A51, 0x11A5B}, {0x11A8A, 0x11A99},
    {0x11B60, 0x11B67}, {0x11C2F, 0x11C36}, {0x11C38, 0x11C3F}, {0x11C92, 0x11CA7}, {0x11CA9, 0x11CB6},
    {0x11D31, 0x11D36}, {0x11D3A, 0x11D3A}, {0x11D3C, 0x11D3D}, {0x11D3F, 0x11D45}, {0x11D47, 0x11D47},
    {0x11D8A, 0x11D8E}, {0x11D90, 0x11D91}, {0x11D93, 0x11D97}, {0x11EF3, 0x11EF6}, {0x11F00, 0x11F01},
    {0x11F03, 0x11F03}, {0x11F34, 0x11F3A}, {0x11F3E, 0x11F42}, {0x11F5A, 0x11F5A}, {0x13440, 0x13440},
    {0x13447, 0x13455}, {0x1611E, 0x1612F}, {0x16AF0, 0x16AF4}, {0x16B30, 0x16B36}, {0x16F4F, 0x16F4F},
    {0x16F51, 0x16F87}, {0x16F8F, 0x16F92}, {0x16FE4, 0x16FE4}, {0x16FF0, 0x16FF1}, {0x1BC9D, 0x1BC9E},
    {0x1CF00, 0x1CF2D}, {0x1CF30, 0x1CF46}, {0x1D165, 0x1D169}, {0x1D16D, 0x1D172}, {0x1D17B, 0x1D182},
    {0x1D185, 0x1D18B}, {0x1D1AA, 0x1D1AD}, {0x1D242, 0x1D244}, {0x1DA00, 0x1DA36}, {0x1DA3B, 0x1DA6C},
    {0x1DA75, 0x1DA75}, {0x1DA84, 0x1DA84}, {0x1DA9B, 0x1DA9F}, {0x1DAA1, 0x1DAAF}, {0x1E000, 0x1E006},
    {0x1E008, 0x1E018}, {0x1E01B, 0x1E021}, {0x1E023, 0x1E024}, {0x1E026, 0x1E02A}, {0x1E08F, 0x1E08F},
    {0x1E130, 0x1E136}, {0x1E2AE, 0x1E2AE}, {0x1E2EC, 0x1E2EF}, {0x1E4EC, 0x1E4EF}, {0x1E5EE, 0x1E5EF},
    {0x1E6E3, 0x1E6E3}, {0x1E6E6, 0x1E6E6}, {0x1E6EE, 0x1E6EF}, {0x1E6F5, 0x1E6F5}, {0x1E8D0, 0x1E8D6},
    {0x1E944, 0x1E94A}, {0xE0100, 0xE01EF},
};

static int is_mark(uint32_t cp) {
  size_t lo = 0, hi = sizeof(MARK_RANGES) / sizeof(MARK_RANGES[0]);
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (cp < MARK_RANGES[mid][0]) hi = mid;
    else if (cp > MARK_RANGES[mid][1]) lo = mid + 1;
    else return 1;
  }
  return 0;
}

/* ------------------------------------------------------------- slugify */

typedef struct {
  Buf out;
  int gap;
} SlugCtx;

static void slug_feed(SlugCtx *c, uint32_t cp, int depth);

static void slug_feed_digits(SlugCtx *c, unsigned v, int depth) {
  if (v >= 10) slug_feed(c, (uint32_t)('0' + v / 10), depth);
  slug_feed(c, (uint32_t)('0' + v % 10), depth);
}

static void slug_feed(SlugCtx *c, uint32_t cp, int depth) {
  /* Formula decompositions: fullwidth, enclosed alphanumerics. */
  if (cp >= 0xFF01 && cp <= 0xFF5E) {
    slug_feed(c, cp - 0xFEE0, depth + 1);
    return;
  }
  if (cp >= 0x2460 && cp <= 0x2473) {
    slug_feed_digits(c, cp - 0x245F, depth + 1);
    return;
  }
  if (cp >= 0x2474 && cp <= 0x2487) {
    slug_feed(c, '(', depth + 1);
    slug_feed_digits(c, cp - 0x2473, depth + 1);
    slug_feed(c, ')', depth + 1);
    return;
  }
  if (cp >= 0x2488 && cp <= 0x249B) {
    slug_feed_digits(c, cp - 0x2487, depth + 1);
    slug_feed(c, '.', depth + 1);
    return;
  }
  if (cp >= 0x249C && cp <= 0x24B5) {
    slug_feed(c, '(', depth + 1);
    slug_feed(c, (uint32_t)('a' + cp - 0x249C), depth + 1);
    slug_feed(c, ')', depth + 1);
    return;
  }
  if (cp >= 0x24B6 && cp <= 0x24CF) {
    slug_feed(c, 'A' + (cp - 0x24B6), depth + 1);
    return;
  }
  if (cp >= 0x24D0 && cp <= 0x24E9) {
    slug_feed(c, 'a' + (cp - 0x24D0), depth + 1);
    return;
  }
  if (cp == 0x24EA) {
    slug_feed(c, '0', depth + 1);
    return;
  }
  if (cp == 0x1F100) {
    slug_feed(c, '0', depth + 1);
    slug_feed(c, '.', depth + 1);
    return;
  }
  if (cp >= 0x1F101 && cp <= 0x1F10A) {
    slug_feed_digits(c, cp - 0x1F101, depth + 1);
    slug_feed(c, ',', depth + 1);
    return;
  }
  if (cp >= 0x1F110 && cp <= 0x1F129) {
    slug_feed(c, '(', depth + 1);
    slug_feed(c, (uint32_t)('A' + cp - 0x1F110), depth + 1);
    slug_feed(c, ')', depth + 1);
    return;
  }
  if (cp >= 0x1F130 && cp <= 0x1F149) {
    slug_feed(c, (uint32_t)('A' + cp - 0x1F130), depth + 1);
    return;
  }
  /* Mathematical alphanumeric symbols: structured 26-letter runs. */
  if (cp >= 0x1D400 && cp <= 0x1D433) { /* bold */
    unsigned o = cp - 0x1D400;
    slug_feed(c, (uint32_t)((o < 26 ? 'A' : 'a') + o % 26), depth + 1);
    return;
  }
  if (cp >= 0x1D434 && cp <= 0x1D467 && cp != 0x1D455) { /* italic; h is Planck */
    unsigned o = cp - 0x1D434;
    slug_feed(c, (uint32_t)((o < 26 ? 'A' : 'a') + o % 26), depth + 1);
    return;
  }
  if (cp >= 0x1D468 && cp <= 0x1D49B) { /* bold italic */
    unsigned o = cp - 0x1D468;
    slug_feed(c, (uint32_t)((o < 26 ? 'A' : 'a') + o % 26), depth + 1);
    return;
  }
  if (cp >= 0x1D49C && cp <= 0x1D4CF) { /* script; holes map to letterlikes */
    unsigned o = cp - 0x1D49C;
    int hole = (o == 1 || o == 4 || o == 5 || o == 7 || o == 8 || o == 11 ||
                o == 12 || o == 17 || o == 30 || o == 32 || o == 40);
    if (!hole) {
      slug_feed(c, (uint32_t)((o < 26 ? 'A' : 'a') + o % 26), depth + 1);
      return;
    }
  }
  if (cp >= 0x1D4D0 && cp <= 0x1D503) { /* fraktur */
    unsigned o = cp - 0x1D4D0;
    slug_feed(c, (uint32_t)((o < 26 ? 'A' : 'a') + o % 26), depth + 1);
    return;
  }
  if (cp >= 0x1D504 && cp <= 0x1D537) { /* double-struck; holes map to letterlikes */
    unsigned o = cp - 0x1D504;
    int hole = (o == 2 || o == 7 || o == 8 || o == 17 || o == 25);
    if (!hole) {
      slug_feed(c, (uint32_t)((o < 26 ? 'A' : 'a') + o % 26), depth + 1);
      return;
    }
  }
  if (cp >= 0x1D538 && cp <= 0x1D551) { /* bold fraktur; holes map to letterlikes */
    unsigned o = cp - 0x1D538;
    int hole = (o == 2 || o == 7 || o == 13 || o == 15 || o == 16 || o == 17 || o == 25);
    if (!hole) {
      slug_feed(c, (uint32_t)((o < 26 ? 'A' : 'a') + o % 26), depth + 1);
      return;
    }
  }
  if (cp >= 0x1D552 && cp <= 0x1D6A3) { /* sans/italic/mono runs, alternating case */
    unsigned idx = cp - 0x1D552;
    unsigned run = idx / 26, pos = idx % 26;
    slug_feed(c, (uint32_t)((run & 1) ? 'A' + pos : 'a' + pos), depth + 1);
    return;
  }
  if (cp >= 0x1D7CE && cp <= 0x1D7FF) { /* math digits, five styles of 0-9 */
    slug_feed_digits(c, (cp - 0x1D7CE) % 10, depth + 1);
    return;
  }
  if (cp >= 0x1CCD6 && cp <= 0x1CCEF) { /* squared latin capitals */
    slug_feed(c, (uint32_t)('A' + cp - 0x1CCD6), depth + 1);
    return;
  }
  if (cp >= 0x1CCF0 && cp <= 0x1CCF9) { /* squared digits */
    slug_feed(c, (uint32_t)('0' + cp - 0x1CCF0), depth + 1);
    return;
  }
  if (cp >= 0x1FBF0 && cp <= 0x1FBF9) { /* segmented digits */
    slug_feed(c, (uint32_t)('0' + cp - 0x1FBF0), depth + 1);
    return;
  }
  if (cp >= 0x3358 && cp <= 0x3370) { /* squared day numbers + ideograph */
    slug_feed_digits(c, cp - 0x3358, depth + 1);
    slug_feed(c, 0x70B9, depth + 1);
    return;
  }
  if (cp >= 0x33E0 && cp <= 0x33FE) { /* squared day numbers + ideograph */
    slug_feed_digits(c, cp - 0x33DF, depth + 1);
    slug_feed(c, 0x65E5, depth + 1);
    return;
  }
  if (cp == 0x33AE) { /* rad/s */
    slug_feed(c, 'r', depth + 1);
    slug_feed(c, 'a', depth + 1);
    slug_feed(c, 'd', depth + 1);
    slug_feed(c, 0x2215, depth + 1);
    slug_feed(c, 's', depth + 1);
    return;
  }
  if (cp == 0x33AF) { /* rad/s squared */
    slug_feed(c, 'r', depth + 1);
    slug_feed(c, 'a', depth + 1);
    slug_feed(c, 'd', depth + 1);
    slug_feed(c, 0x2215, depth + 1);
    slug_feed(c, 's', depth + 1);
    slug_feed(c, '2', depth + 1);
    return;
  }
  if (depth < 4) {
    const Decomp *d = decomp_find(cp);
    if (d) {
      for (int i = 0; i < d->n; i++) slug_feed(c, d->d[i], depth + 1);
      return;
    }
  }
  if (is_mark(cp)) return;
  if (cp >= 'A' && cp <= 'Z') cp += 32;
  if ((cp >= 'a' && cp <= 'z') || (cp >= '0' && cp <= '9')) {
    if (c->gap && c->out.len > 0) buf_addc(&c->out, '-');
    c->gap = 0;
    buf_addc(&c->out, (unsigned char)cp);
  } else {
    c->gap = c->out.len > 0;
  }
}

static Str slugify(Arena *ar, Str s) {
  SlugCtx c;
  c.out.b = NULL;
  c.out.len = c.out.cap = 0;
  c.gap = 0;
  const unsigned char *p = s.p, *e = s.p + s.n;
  while (p < e) {
    uint32_t cp;
    int n = u8dec(p, e, &cp);
    if (n < 0) {
      cp = *p;
      n = 1;
    }
    slug_feed(&c, cp, 0);
    p += n;
  }
  size_t n = c.out.len > 64 ? 64 : c.out.len;
  while (n > 0 && c.out.b[n - 1] == '-') n--;
  Str r;
  if (n == 0) r = str_from_c("untitled");
  else r = ar_str(ar, c.out.b, n);
  buf_free(&c.out);
  return r;
}

/* First of slug, slug-2, slug-3, ... not yet in `taken` (and added to it). */
static Str unique_slug(Arena *ar, Str slug, SMap *taken) {
  if (!smap_has(taken, slug)) {
    smap_put(taken, slug, 1);
    return slug;
  }
  char tmp[96];
  for (int k = 2;; k++) {
    int m = snprintf(tmp, sizeof(tmp), "-%d", k);
    size_t n = slug.n + (size_t)m;
    unsigned char *b = (unsigned char *)ar_alloc(ar, n ? n : 1);
    memcpy(b, slug.p, slug.n);
    memcpy(b + slug.n, tmp, (size_t)m);
    Str cand = {b, n};
    if (!smap_has(taken, cand)) {
      smap_put(taken, cand, 1);
      return cand;
    }
  }
}

/* ------------------------------------------------------------ HTML escape */

static int have_avx2(void) {
#if defined(__x86_64__)
  static int v = -1;
  if (v < 0) v = __builtin_cpu_supports("avx2") ? 1 : 0;
  return v;
#else
  return 0;
#endif
}

#if defined(__x86_64__)
/* Bitmask of the 32 bytes at p holding one of & < > " '. */
__attribute__((target("avx2"))) static int chunk_special_mask_avx2(const unsigned char *p) {
  const __m256i amp = _mm256_set1_epi8('&');
  const __m256i lt = _mm256_set1_epi8('<');
  const __m256i gt = _mm256_set1_epi8('>');
  const __m256i qo = _mm256_set1_epi8('"');
  const __m256i ap = _mm256_set1_epi8('\'');
  __m256i v = _mm256_loadu_si256((const __m256i *)p);
  __m256i m = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(v, amp), _mm256_cmpeq_epi8(v, lt)),
                              _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(v, gt), _mm256_cmpeq_epi8(v, qo)),
                                             _mm256_cmpeq_epi8(v, ap)));
  return _mm256_movemask_epi8(m);
}
#endif

static const unsigned char *find_special(const unsigned char *p, const unsigned char *e) {
  if (have_avx2()) {
#if defined(__x86_64__)
    while (e - p >= 32) {
      int mask = chunk_special_mask_avx2(p);
      if (mask) return p + __builtin_ctz((unsigned)mask);
      p += 32;
    }
#endif
  }
  while (p < e) {
    unsigned char c = *p;
    if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') return p;
    p++;
  }
  return NULL;
}

#if defined(__x86_64__)
/* Bitmask of the 32 bytes at p holding a byte that needs a JSON escape:
 * '"' '\\', any control < 0x20, or 0xED (possible lone-surrogate lead). */
__attribute__((target("avx2"))) static int chunk_json_mask_avx2(const unsigned char *p) {
  const __m256i qo = _mm256_set1_epi8('"');
  const __m256i bs = _mm256_set1_epi8('\\');
  const __m256i ed = _mm256_set1_epi8((char)0xED);
  const __m256i sp = _mm256_set1_epi8(' ');
  const __m256i zero = _mm256_setzero_si256();
  __m256i v = _mm256_loadu_si256((const __m256i *)p);
  __m256i m = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(v, qo), _mm256_cmpeq_epi8(v, bs)),
                              _mm256_cmpeq_epi8(v, ed));
  __m256i t = _mm256_subs_epu8(v, sp);
  __m256i low = _mm256_andnot_si256(_mm256_cmpeq_epi8(v, sp), _mm256_cmpeq_epi8(t, zero));
  return _mm256_movemask_epi8(_mm256_or_si256(m, low));
}

/* Bitmask of the 32 bytes at p holding an inline-markdown or HTML-escape char:
 * backslash, backtick, star, '[', '&', '<', '>', '"', '\''. */
__attribute__((target("avx2"))) static int chunk_inline_mask_avx2(const unsigned char *p) {
  const __m256i bs = _mm256_set1_epi8('\\');
  const __m256i bt = _mm256_set1_epi8('`');
  const __m256i st = _mm256_set1_epi8('*');
  const __m256i ob = _mm256_set1_epi8('[');
  const __m256i amp = _mm256_set1_epi8('&');
  const __m256i lt = _mm256_set1_epi8('<');
  const __m256i gt = _mm256_set1_epi8('>');
  const __m256i qo = _mm256_set1_epi8('"');
  const __m256i ap = _mm256_set1_epi8('\'');
  __m256i v = _mm256_loadu_si256((const __m256i *)p);
  __m256i m = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(v, bs), _mm256_cmpeq_epi8(v, bt)),
                              _mm256_or_si256(_mm256_cmpeq_epi8(v, st), _mm256_cmpeq_epi8(v, ob)));
  m = _mm256_or_si256(m, _mm256_or_si256(_mm256_cmpeq_epi8(v, amp), _mm256_cmpeq_epi8(v, lt)));
  m = _mm256_or_si256(m, _mm256_or_si256(_mm256_cmpeq_epi8(v, gt), _mm256_cmpeq_epi8(v, qo)));
  m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, ap));
  return _mm256_movemask_epi8(m);
}
#endif

/* First byte at or after p (before e) that needs a JSON escape, or NULL. */
static const unsigned char *find_json_special(const unsigned char *p, const unsigned char *e) {
  if (have_avx2()) {
#if defined(__x86_64__)
    while (e - p >= 32) {
      int mask = chunk_json_mask_avx2(p);
      if (mask) return p + __builtin_ctz((unsigned)mask);
      p += 32;
    }
#endif
  }
  while (p < e) {
    unsigned char c = *p;
    if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) return p;
    p++;
  }
  return NULL;
}

/* First byte at or after p (before e) that is an inline-markdown or
 * HTML-escape char, or NULL. */
static const unsigned char *find_inline_special(const unsigned char *p, const unsigned char *e) {
  if (have_avx2()) {
#if defined(__x86_64__)
    while (e - p >= 32) {
      int mask = chunk_inline_mask_avx2(p);
      if (mask) return p + __builtin_ctz((unsigned)mask);
      p += 32;
    }
#endif
  }
  while (p < e) {
    unsigned char c = *p;
    if (c == '\\' || c == '`' || c == '*' || c == '[' || c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') return p;
    p++;
  }
  return NULL;
}

/* Append `s` with & < > " ' escaped, exactly like escapeHtml. */
static void esc_append(Buf *out, Str s) {
  const unsigned char *p = s.p, *e = s.p + s.n;
  while (p < e) {
    const unsigned char *q = find_special(p, e);
    if (!q) {
      buf_addn(out, p, (size_t)(e - p));
      return;
    }
    if (q > p) buf_addn(out, p, (size_t)(q - p));
    switch (*q) {
      case '&': buf_adds(out, "&amp;"); break;
      case '<': buf_adds(out, "&lt;"); break;
      case '>': buf_adds(out, "&gt;"); break;
      case '"': buf_adds(out, "&quot;"); break;
      default: buf_adds(out, "&#39;"); break;
    }
    p = q + 1;
  }
}

/* Escape one ASCII character (used for backslash escapes in inline code). */
static void esc_append_char(Buf *out, unsigned char c) {
  switch (c) {
    case '&': buf_adds(out, "&amp;"); break;
    case '<': buf_adds(out, "&lt;"); break;
    case '>': buf_adds(out, "&gt;"); break;
    case '"': buf_adds(out, "&quot;"); break;
    case '\'': buf_adds(out, "&#39;"); break;
    default: buf_addc(out, c); break;
  }
}

/* ------------------------------------------------------------ JSON escape */

/* Append `s` escaped the way JSON.stringify escapes a string's contents
 * (no surrounding quotes). Lone surrogates (WTF-8) become \udXXX. */
static void json_escape_into(Buf *out, Str s) {
  static const char hex[] = "0123456789abcdef";
  const unsigned char *p = s.p, *e = s.p + s.n;
  while (p < e) {
    const unsigned char *q = find_json_special(p, e);
    if (!q) {
      buf_addn(out, p, (size_t)(e - p));
      return;
    }
    if (q > p) buf_addn(out, p, (size_t)(q - p));
    unsigned char c = *q;
    if (c == 0xED && e - q >= 3 && q[1] >= 0xA0 && q[1] <= 0xBF && (q[2] & 0xC0) == 0x80) {
      uint32_t cp = ((uint32_t)(q[0] & 0x0F) << 12) | ((uint32_t)(q[1] & 0x3F) << 6) | (q[2] & 0x3F);
      char tmp[8];
      tmp[0] = '\\';
      tmp[1] = 'u';
      tmp[2] = hex[(cp >> 12) & 15];
      tmp[3] = hex[(cp >> 8) & 15];
      tmp[4] = hex[(cp >> 4) & 15];
      tmp[5] = hex[cp & 15];
      buf_addn(out, tmp, 6);
      p = q + 3;
      continue;
    }
    switch (c) {
      case '"': buf_adds(out, "\\\""); break;
      case '\\': buf_adds(out, "\\\\"); break;
      case '\b': buf_adds(out, "\\b"); break;
      case '\f': buf_adds(out, "\\f"); break;
      case '\n': buf_adds(out, "\\n"); break;
      case '\r': buf_adds(out, "\\r"); break;
      case '\t': buf_adds(out, "\\t"); break;
      default: {
        if (c < 0x20) {
          char tmp[8];
          tmp[0] = '\\';
          tmp[1] = 'u';
          tmp[2] = '0';
          tmp[3] = '0';
          tmp[4] = hex[(c >> 4) & 15];
          tmp[5] = hex[c & 15];
          buf_addn(out, tmp, 6);
        } else {
          /* 0xED that is not a lone-surrogate lead: copy the byte raw. */
          buf_addc(out, c);
        }
        break;
      }
    }
    p = q + 1;
  }
}

static void json_quote(Buf *out, Str s) {
  buf_addc(out, '"');
  json_escape_into(out, s);
  buf_addc(out, '"');
}

/* ----------------------------------------------------------- front matter */

typedef struct {
  Str title, date, layout, slug, summary;
  Str *tags;
  int ntags;
  int has_date, has_layout, has_slug, has_summary, draft;
  Str body;
} FM;

static int days_in(int y, int m) {
  if (m == 2) return ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 29 : 28;
  return (m == 4 || m == 6 || m == 9 || m == 11) ? 30 : 31;
}

static int is_blank_line(Str l) {
  for (size_t i = 0; i < l.n; i++)
    if (l.p[i] != ' ' && l.p[i] != '\t') return 0;
  return 1;
}

/* A quoted or bare string value; on error returns -1 and sets *err. */
static int fm_string(Arena *ar, Str raw, Str *out, const char **err) {
  if (raw.n > 0 && raw.p[0] == '"') {
    if (raw.n < 2 || raw.p[raw.n - 1] != '"') {
      *err = "a quoted string is not closed";
      return -1;
    }
    Buf u = {0};
    for (size_t i = 1; i + 1 < raw.n; i++) {
      unsigned char c = raw.p[i];
      if (c == '\\') {
        if (i + 1 >= raw.n - 1) {
          buf_free(&u);
          *err = "a quoted string holds an escape other than \\\" or \\\\";
          return -1;
        }
        unsigned char n = raw.p[i + 1];
        if (n != '"' && n != '\\') {
          buf_free(&u);
          *err = "a quoted string holds an escape other than \\\" or \\\\";
          return -1;
        }
        buf_addc(&u, n);
        i++;
      } else if (c == '"') {
        buf_free(&u);
        *err = "a quoted string holds a stray quote";
        return -1;
      } else {
        buf_addc(&u, c);
      }
    }
    *out = buf_to_str(ar, &u);
    buf_free(&u);
    return 0;
  }
  if (raw.n > 0 && raw.p[0] == '[') {
    *err = "a bare string opens on [";
    return -1;
  }
  *out = raw;
  return 0;
}

/* Parse front matter. Returns 0 ok, -1 refused with *err (arena string). */
static int parse_fm(Arena *ar, Str text, FM *fm, Str *err) {
  memset(fm, 0, sizeof(*fm));
  /* Split into lines. */
  Str *lines = NULL;
  int nlines = 0, caplines = 0;
  const unsigned char *p = text.p, *e = text.p + text.n;
  while (p <= e) {
    const unsigned char *nl = (const unsigned char *)memchr(p, '\n', (size_t)(e - p));
    const unsigned char *le = nl ? nl : e;
    if (nlines == caplines) {
      int nc = caplines ? caplines * 2 : 16;
      lines = (Str *)ar_grow_copy(ar, lines, (size_t)nlines * sizeof(Str), (size_t)nc * sizeof(Str));
      caplines = nc;
    }
    lines[nlines].p = p;
    lines[nlines].n = (size_t)(le - p);
    nlines++;
    if (!nl) break;
    p = nl + 1;
    if (p > e) break;
    if (p == e) {
      /* Trailing newline: text.split leaves one final empty line. */
      if (nlines == caplines) {
        int nc = caplines ? caplines * 2 : 16;
        lines = (Str *)ar_grow_copy(ar, lines, (size_t)nlines * sizeof(Str), (size_t)nc * sizeof(Str));
        caplines = nc;
      }
      lines[nlines].p = p;
      lines[nlines].n = 0;
      nlines++;
      break;
    }
  }
  if (nlines == 0 || !str_eq_c(lines[0], "---")) {
    *err = str_from_c("no front matter");
    return -1;
  }
  int close = -1;
  for (int i = 1; i < nlines; i++) {
    if (str_eq_c(lines[i], "---")) {
      close = i;
      break;
    }
  }
  if (close == -1) {
    *err = str_from_c("the front matter is not closed");
    return -1;
  }
  static const char *KEYS[] = {"title", "date", "tags", "draft", "layout", "slug", "summary"};
  int seen[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int i = 1; i < close; i++) {
    Str line = lines[i];
    if (is_blank_line(line)) continue;
    if (line.n > 0 && line.p[0] == '#') continue;
    /* /^([a-z_]+):(?: (.*))?$/ */
    size_t k = 0;
    while (k < line.n && ((line.p[k] >= 'a' && line.p[k] <= 'z') || line.p[k] == '_')) k++;
    Str raw = {line.p, 0};
    Str key;
    if (k == 0 || k >= line.n || line.p[k] != ':') goto not_kv;
    key.p = line.p;
    key.n = k;
    if (k + 1 == line.n) {
      raw.p = line.p + k + 1;
      raw.n = 0;
    } else if (line.p[k + 1] == ' ') {
      raw.p = line.p + k + 2;
      raw.n = line.n - (k + 2);
    } else {
      goto not_kv;
    }
    /* Strip trailing spaces and tabs from the value. */
    while (raw.n > 0 && (raw.p[raw.n - 1] == ' ' || raw.p[raw.n - 1] == '\t')) raw.n--;
    {
      int ki = -1;
      for (int t = 0; t < 7; t++)
        if (str_eq_c(key, KEYS[t])) ki = t;
      if (ki < 0) {
        Buf b = {0};
        buf_adds(&b, "unknown key \"");
        buf_addn(&b, key.p, key.n);
        buf_addc(&b, '"');
        *err = buf_to_str(ar, &b);
        buf_free(&b);
        return -1;
      }
      if (seen[ki]) {
        Buf b = {0};
        buf_adds(&b, "key \"");
        buf_addn(&b, key.p, key.n);
        buf_adds(&b, "\" is given twice");
        *err = buf_to_str(ar, &b);
        buf_free(&b);
        return -1;
      }
      seen[ki] = 1;
      if (raw.n == 0) {
        Buf b = {0};
        buf_adds(&b, "key \"");
        buf_addn(&b, key.p, key.n);
        buf_adds(&b, "\" has no value");
        *err = buf_to_str(ar, &b);
        buf_free(&b);
        return -1;
      }
      const char *verr = NULL;
      switch (ki) {
        case 0: /* title */
        case 6: /* summary */ {
          Str v;
          if (fm_string(ar, raw, &v, &verr) != 0) {
            Buf b = {0};
            buf_adds(&b, "key \"");
            buf_addn(&b, key.p, key.n);
            buf_adds(&b, "\": ");
            buf_adds(&b, verr);
            *err = buf_to_str(ar, &b);
            buf_free(&b);
            return -1;
          }
          if (ki == 0) fm->title = v;
          else fm->summary = v, fm->has_summary = 1;
          break;
        }
        case 1: { /* date */
          int ok = 0;
          if (raw.n == 10 && raw.p[4] == '-' && raw.p[7] == '-') {
            int dig = 1;
            for (int t = 0; t < 10; t++) {
              if (t == 4 || t == 7) continue;
              if (raw.p[t] < '0' || raw.p[t] > '9') dig = 0;
            }
            if (dig) {
              int y = (raw.p[0] - '0') * 1000 + (raw.p[1] - '0') * 100 + (raw.p[2] - '0') * 10 + (raw.p[3] - '0');
              int mo = (raw.p[5] - '0') * 10 + (raw.p[6] - '0');
              int d = (raw.p[8] - '0') * 10 + (raw.p[9] - '0');
              if (y >= 1970 && y <= 2099 && mo >= 1 && mo <= 12 && d >= 1 && d <= days_in(y, mo)) ok = 1;
            }
          }
          if (!ok) {
            *err = str_from_c("key \"date\": not a date from 1970-01-01 to 2099-12-31 (YYYY-MM-DD)");
            return -1;
          }
          fm->date = raw;
          fm->has_date = 1;
          break;
        }
        case 2: { /* tags */
          if (raw.n < 2 || raw.p[0] != '[' || raw.p[raw.n - 1] != ']') {
            *err = str_from_c("key \"tags\": not a list [a, b]");
            return -1;
          }
          Str inner = {raw.p + 1, raw.n - 2};
          if (trim_str(inner).n == 0) {
            fm->tags = NULL;
            fm->ntags = 0;
            break;
          }
          /* Count commas for the tag array. */
          int cnt = 1;
          for (size_t t = 0; t < inner.n; t++)
            if (inner.p[t] == ',') cnt++;
          Str *tags = (Str *)ar_alloc(ar, (size_t)cnt * sizeof(Str));
          int nt = 0;
          size_t start = 0;
          for (size_t t = 0; t <= inner.n; t++) {
            if (t == inner.n || inner.p[t] == ',') {
              Str seg = {inner.p + start, t - start};
              Str tr = trim_str(seg);
              if (tr.n == 0) {
                *err = str_from_c("key \"tags\": an empty tag, or one holding \" [ or ]");
                return -1;
              }
              for (size_t u = 0; u < tr.n; u++) {
                if (tr.p[u] == '"' || tr.p[u] == '[' || tr.p[u] == ']') {
                  *err = str_from_c("key \"tags\": an empty tag, or one holding \" [ or ]");
                  return -1;
                }
              }
              tags[nt++] = tr;
              start = t + 1;
            }
          }
          fm->tags = tags;
          fm->ntags = nt;
          break;
        }
        case 3: /* draft */
          if (str_eq_c(raw, "true")) fm->draft = 1;
          else if (str_eq_c(raw, "false")) fm->draft = 0;
          else {
            *err = str_from_c("key \"draft\": true or false");
            return -1;
          }
          break;
        case 4: { /* layout */
          int ok = raw.n > 0;
          for (size_t t = 0; ok && t < raw.n; t++) {
            unsigned char ch = raw.p[t];
            if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-')) ok = 0;
          }
          if (!ok) {
            *err = str_from_c("key \"layout\": not a template name");
            return -1;
          }
          fm->layout = raw;
          fm->has_layout = 1;
          break;
        }
        case 5: { /* slug */
          int ok = raw.n > 0 && raw.n <= 64;
          if (ok) {
            size_t t = 0;
            /* /^[a-z0-9]+(?:-[a-z0-9]+)*$/ */
            while (t < raw.n) {
              if (!((raw.p[t] >= 'a' && raw.p[t] <= 'z') || (raw.p[t] >= '0' && raw.p[t] <= '9'))) {
                ok = 0;
                break;
              }
              while (t < raw.n && ((raw.p[t] >= 'a' && raw.p[t] <= 'z') || (raw.p[t] >= '0' && raw.p[t] <= '9'))) t++;
              if (t < raw.n) {
                if (raw.p[t] != '-') {
                  ok = 0;
                  break;
                }
                t++;
                if (t >= raw.n) {
                  ok = 0;
                  break;
                }
              }
            }
          }
          if (!ok) {
            *err = str_from_c("key \"slug\": not a slug");
            return -1;
          }
          fm->slug = raw;
          fm->has_slug = 1;
          break;
        }
      }
    }
    continue;
  not_kv: {
    char num[16];
    int m = snprintf(num, sizeof(num), "%d", i + 1);
    Buf b = {0};
    buf_adds(&b, "line ");
    buf_addn(&b, num, (size_t)m);
    buf_adds(&b, " is not \"key: value\"");
    *err = buf_to_str(ar, &b);
    buf_free(&b);
    return -1;
  }
  }
  if (!seen[0]) {
    *err = str_from_c("no title");
    return -1;
  }
  /* Body: everything after the closing line's newline (or empty). */
  {
    const unsigned char *bp;
    size_t bl;
    /* Find the byte offset just past the close line. */
    const unsigned char *q = text.p;
    const unsigned char *e2 = text.p + text.n;
    for (int i = 0; i <= close; i++) {
      const unsigned char *nl = (const unsigned char *)memchr(q, '\n', (size_t)(e2 - q));
      if (!nl) {
        q = e2;
        break;
      }
      q = nl + 1;
    }
    bp = q;
    bl = (size_t)(e2 - q);
    fm->body.p = bp;
    fm->body.n = bl;
  }
  return 0;
}

/* ------------------------------------------------------- build structures */

typedef struct Val Val;

typedef struct Source Source;
typedef struct Tag Tag;

typedef struct {
  int level;
  Str id, html;
} TocEnt;

struct Tag {
  Str name, slug, url;
  Source **posts;
  int nposts, cap;
};

struct Source {
  Str path, text;
  Str title, date, layout, slug, summary;
  Str *tags;
  int ntags;
  int has_date, has_layout, has_slug, has_summary, draft;
  Str body;
  int post;
  Str slugf, outdir, url;
  Str html, excerpt;
  TocEnt *toc;
  int ntoc;
  long long read_min;
  Str date_long;
  Source *newer, *older;
  Tag **tagrefs;
  int ntagrefs;
  Val *card;
};

typedef struct {
  Arena *ar;
  Source **srcs;
  SMap *by_src; /* path -> index into srcs */
  SMap *statics; /* set of static paths */
  Str base;
  Str cur_path, cur_dir;
  SMap *broken_set;
  Str *broken;
  int nbroken, capbroken;
} MdCtx;

static void split_lines(Arena *ar, Str text, Str **out_lines, int *out_n) {
  Str *lines = NULL;
  int n = 0, cap = 0;
  const unsigned char *p = text.p, *e = text.p + text.n;
  while (p <= e) {
    const unsigned char *nl = (const unsigned char *)memchr(p, '\n', (size_t)(e - p));
    const unsigned char *le = nl ? nl : e;
    if (n == cap) {
      int nc = cap ? cap * 2 : 16;
      lines = (Str *)ar_grow_copy(ar, lines, (size_t)n * sizeof(Str), (size_t)nc * sizeof(Str));
      cap = nc;
    }
    lines[n].p = p;
    lines[n].n = (size_t)(le - p);
    n++;
    if (!nl) break;
    p = nl + 1;
    if (p == e) {
      if (n == cap) {
        int nc = cap ? cap * 2 : 16;
        lines = (Str *)ar_grow_copy(ar, lines, (size_t)n * sizeof(Str), (size_t)nc * sizeof(Str));
        cap = nc;
      }
      lines[n].p = p;
      lines[n].n = 0;
      n++;
      break;
    }
    if (p > e) break;
  }
  *out_lines = lines;
  *out_n = n;
}

static long long words_of(Str s) {
  long long n = 0;
  int in_word = 0;
  const unsigned char *p = s.p, *e = s.p + s.n;
  while (p < e) {
    int sp;
    if (*p < 0x80) {
      sp = js_space(*p);
      p++;
    } else {
      uint32_t cp;
      int d = u8dec(p, e, &cp);
      if (d < 0) {
        sp = 0;
        p++;
      } else {
        sp = js_space(cp);
        p += d;
      }
    }
    if (!sp && !in_word) {
      n++;
      in_word = 1;
    } else if (sp) {
      in_word = 0;
    }
  }
  return n;
}

/* ---------------------------------------------------------------- markdown */

static int is_blank_line2(Str l) {
  for (size_t i = 0; i < l.n; i++)
    if (l.p[i] != ' ' && l.p[i] != '\t') return 0;
  return 1;
}

static Str trim_line(Str l) {
  size_t a = 0, b = l.n;
  while (a < b && (l.p[a] == ' ' || l.p[a] == '\t')) a++;
  while (b > a && (l.p[b - 1] == ' ' || l.p[b - 1] == '\t')) b--;
  Str r = {l.p + a, b - a};
  return r;
}

static int is_lang_char(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '-';
}

static int fence_open(Str l, Str *lang) {
  if (l.n < 3 || memcmp(l.p, "```", 3) != 0) return 0;
  size_t k = 3;
  while (k < l.n && is_lang_char(l.p[k])) k++;
  for (size_t t = k; t < l.n; t++)
    if (l.p[t] != ' ' && l.p[t] != '\t') return 0;
  lang->p = l.p + 3;
  lang->n = k - 3;
  return 1;
}

static int fence_close(Str l) {
  if (l.n < 3 || memcmp(l.p, "```", 3) != 0) return 0;
  for (size_t t = 3; t < l.n; t++)
    if (l.p[t] != ' ' && l.p[t] != '\t') return 0;
  return 1;
}

static int heading_of(Str l, int *depth, Str *text) {
  size_t k = 0;
  while (k < l.n && l.p[k] == '#') k++;
  if (k < 1 || k > 6) return 0;
  if (k >= l.n || (l.p[k] != ' ' && l.p[k] != '\t')) return 0;
  size_t s = k;
  while (s < l.n && (l.p[s] == ' ' || l.p[s] == '\t')) s++;
  size_t e = l.n;
  while (e > s && (l.p[e - 1] == ' ' || l.p[e - 1] == '\t')) e--;
  if (e == s) return 0;
  *depth = (int)k;
  text->p = l.p + s;
  text->n = e - s;
  return 1;
}

static int is_hr(Str l) {
  if (l.n < 3) return 0;
  int dash = l.p[0] == '-' && l.p[1] == '-' && l.p[2] == '-';
  int star = l.p[0] == '*' && l.p[1] == '*' && l.p[2] == '*';
  if (!dash && !star) return 0;
  for (size_t t = 3; t < l.n; t++)
    if (l.p[t] != ' ' && l.p[t] != '\t') return 0;
  return 1;
}

static int match_bullet(Str l, Str *rest) {
  if (l.n < 2 || (l.p[0] != '-' && l.p[0] != '*') || l.p[1] != ' ') return 0;
  rest->p = l.p + 2;
  rest->n = l.n - 2;
  return 1;
}

static int match_ordered(Str l, long *num, Str *rest) {
  size_t k = 0;
  while (k < l.n && l.p[k] >= '0' && l.p[k] <= '9') k++;
  if (k < 1 || k > 9) return 0;
  if (k + 2 > l.n || l.p[k] != '.' || l.p[k + 1] != ' ') return 0;
  long v = 0;
  for (size_t t = 0; t < k; t++) v = v * 10 + (l.p[t] - '0');
  *num = v;
  rest->p = l.p + k + 2;
  rest->n = l.n - (k + 2);
  return 1;
}

static int is_continued(Str l) {
  size_t k = 0;
  while (k < l.n && l.p[k] == ' ') k++;
  if (k < 2 || k >= l.n) return 0;
  return !at_space(l.p + k, l.p + l.n);
}

static int ends_paragraph(Str l) {
  Str lang;
  int d;
  Str t;
  if (is_blank_line2(l)) return 1;
  if (fence_open(l, &lang)) return 1;
  if (heading_of(l, &d, &t)) return 1;
  if (is_hr(l)) return 1;
  return 0;
}

static int is_escapable(unsigned char c) {
  switch (c) {
    case '\\':
    case '`':
    case '*':
    case '_':
    case '[':
    case ']':
    case '(':
    case ')':
    case '#':
    case '+':
    case '-':
    case '.':
    case '!':
    case '>':
      return 1;
    default:
      return 0;
  }
}

static long find_unesc(Str s, const char *what, size_t from) {
  size_t wl = strlen(what);
  for (size_t j = from; j + wl <= s.n; j++) {
    if (s.p[j] == '\\' && j + 1 < s.n && is_escapable(s.p[j + 1])) {
      j++;
      continue;
    }
    if (memcmp(s.p + j, what, wl) == 0) return (long)j;
  }
  return -1;
}

static long find_single_star(Str s, size_t from) {
  for (size_t j = from; j < s.n; j++) {
    unsigned char c = s.p[j];
    if (c == '\\' && j + 1 < s.n && is_escapable(s.p[j + 1])) {
      j++;
      continue;
    }
    if (c != '*') continue;
    if (j + 1 < s.n && s.p[j + 1] == '*') {
      j++;
      continue;
    }
    return (long)j;
  }
  return -1;
}

static int has_ws(Str s) {
  const unsigned char *p = s.p, *e = s.p + s.n;
  while (p < e) {
    if (*p < 0x80) {
      if (js_space(*p)) return 1;
      p++;
    } else {
      uint32_t cp;
      int d = u8dec(p, e, &cp);
      if (d < 0) {
        p++;
        continue;
      }
      if (js_space(cp)) return 1;
      p += d;
    }
  }
  return 0;
}

static int is_scheme(Str t) {
  if (t.n < 2) return 0;
  unsigned char c0 = t.p[0];
  if (!((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z'))) return 0;
  size_t i = 1;
  while (i < t.n) {
    unsigned char c = t.p[i];
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.')
      i++;
    else
      break;
  }
  return i < t.n && t.p[i] == ':';
}

static int resolve_path(Arena *ar, Str dir, Str rel, Str *out) {
  Str *parts = NULL;
  int nparts = 0, capparts = 0;
  if (dir.n > 0) {
    size_t start = 0;
    for (size_t t = 0; t <= dir.n; t++) {
      if (t == dir.n || dir.p[t] == '/') {
        if (nparts == capparts) {
          int nc = capparts ? capparts * 2 : 8;
          parts = (Str *)ar_grow_copy(ar, parts, (size_t)nparts * sizeof(Str), (size_t)nc * sizeof(Str));
          capparts = nc;
        }
        parts[nparts].p = dir.p + start;
        parts[nparts].n = t - start;
        nparts++;
        start = t + 1;
      }
    }
  }
  size_t start = 0;
  for (size_t t = 0; t <= rel.n; t++) {
    if (t == rel.n || rel.p[t] == '/') {
      Str seg = {rel.p + start, t - start};
      if (seg.n == 0 || str_eq_c(seg, ".")) {
        /* skip */
      } else if (str_eq_c(seg, "..")) {
        if (nparts == 0) return 0;
        nparts--;
      } else {
        if (nparts == capparts) {
          int nc = capparts ? capparts * 2 : 8;
          parts = (Str *)ar_grow_copy(ar, parts, (size_t)nparts * sizeof(Str), (size_t)nc * sizeof(Str));
          capparts = nc;
        }
        parts[nparts++] = seg;
      }
      start = t + 1;
    }
  }
  Buf b = {0};
  for (int i = 0; i < nparts; i++) {
    if (i) buf_addc(&b, '/');
    buf_addn(&b, parts[i].p, parts[i].n);
  }
  *out = buf_to_str(ar, &b);
  buf_free(&b);
  return 1;
}

static Str resolve_link(MdCtx *ctx, Str target) {
  if (is_scheme(target) || (target.n > 0 && (target.p[0] == '/' || target.p[0] == '#'))) return target;
  const unsigned char *h = (const unsigned char *)memchr(target.p, '#', target.n);
  Str rel, frag;
  if (h) {
    rel.p = target.p;
    rel.n = (size_t)(h - target.p);
    frag.p = h;
    frag.n = (size_t)(target.p + target.n - h);
  } else {
    rel = target;
    frag.p = target.p + target.n;
    frag.n = 0;
  }
  Str path = {NULL, 0};
  int have_path = 0;
  if (rel.n > 0) have_path = resolve_path(ctx->ar, ctx->cur_dir, rel, &path);
  if (have_path) {
    int idx = smap_get(ctx->by_src, path);
    if (idx >= 0) {
      Buf b = {0};
      buf_addn(&b, ctx->srcs[idx]->url.p, ctx->srcs[idx]->url.n);
      buf_addn(&b, frag.p, frag.n);
      Str r = buf_to_str(ctx->ar, &b);
      buf_free(&b);
      return r;
    }
    if (smap_has(ctx->statics, path)) {
      Buf b = {0};
      buf_addn(&b, ctx->base.p, ctx->base.n);
      buf_addc(&b, '/');
      buf_addn(&b, path.p, path.n);
      buf_addn(&b, frag.p, frag.n);
      Str r = buf_to_str(ctx->ar, &b);
      buf_free(&b);
      return r;
    }
  }
  {
    Buf b = {0};
    buf_adds(&b, "broken-link ");
    buf_addn(&b, ctx->cur_path.p, ctx->cur_path.n);
    buf_adds(&b, " -> ");
    buf_addn(&b, target.p, target.n);
    Str line = buf_to_str(ctx->ar, &b);
    buf_free(&b);
    if (!smap_has(ctx->broken_set, line)) {
      smap_put(ctx->broken_set, line, 1);
      if (ctx->nbroken == ctx->capbroken) {
        int nc = ctx->capbroken ? ctx->capbroken * 2 : 16;
        ctx->broken = (Str *)ar_grow_copy(ctx->ar, ctx->broken, (size_t)ctx->nbroken * sizeof(Str),
                                          (size_t)nc * sizeof(Str));
        ctx->capbroken = nc;
      }
      ctx->broken[ctx->nbroken++] = line;
    }
  }
  return target;
}

static void inline_render(Buf *out, Str s, MdCtx *ctx) {
  size_t i = 0;
  while (i < s.n) {
    /* Fast path: copy the run of plain text up to the next special char. */
    const unsigned char *q = find_inline_special(s.p + i, s.p + s.n);
    if (!q) {
      buf_addn(out, s.p + i, s.n - i);
      break;
    }
    if (q > s.p + i) {
      buf_addn(out, s.p + i, (size_t)(q - (s.p + i)));
      i = (size_t)(q - s.p);
    }
    unsigned char c = s.p[i];
    if (c == '\\' && i + 1 < s.n && is_escapable(s.p[i + 1])) {
      esc_append_char(out, s.p[i + 1]);
      i += 2;
      continue;
    }
    if (c == '\\') {
      buf_addc(out, '\\');
      i++;
      continue;
    }
    if (c == '`') {
      const unsigned char *j = NULL;
      if (i + 1 < s.n) j = (const unsigned char *)memchr(s.p + i + 1, '`', s.n - (i + 1));
      if (j && (size_t)(j - s.p) > i + 1) {
        size_t jj = (size_t)(j - s.p);
        buf_adds(out, "<code>");
        Str inner = {s.p + i + 1, jj - (i + 1)};
        esc_append(out, inner);
        buf_adds(out, "</code>");
        i = jj + 1;
      } else {
        buf_addc(out, '`');
        i++;
      }
      continue;
    }
    if (c == '*') {
      if (i + 1 < s.n && s.p[i + 1] == '*') {
        long j = find_unesc(s, "**", i + 2);
        if (j > (long)(i + 2)) {
          buf_adds(out, "<strong>");
          Str inner = {s.p + i + 2, (size_t)j - (i + 2)};
          inline_render(out, inner, ctx);
          buf_adds(out, "</strong>");
          i = (size_t)j + 2;
        } else {
          buf_adds(out, "**");
          i += 2;
        }
        continue;
      }
      long j = find_single_star(s, i + 1);
      if (j > (long)(i + 1)) {
        buf_adds(out, "<em>");
        Str inner = {s.p + i + 1, (size_t)j - (i + 1)};
        inline_render(out, inner, ctx);
        buf_adds(out, "</em>");
        i = (size_t)j + 1;
      } else {
        buf_addc(out, '*');
        i++;
      }
      continue;
    }
    if (c == '[') {
      long k = find_unesc(s, "]", i + 1);
      long m = -1;
      if (k != -1 && (size_t)k + 1 < s.n && s.p[k + 1] == '(') {
        const unsigned char *r = (const unsigned char *)memchr(s.p + k + 2, ')', s.n - ((size_t)k + 2));
        if (r) m = (long)(r - s.p);
      }
      Str target = {s.p, 0};
      if (m != -1) {
        target.p = s.p + k + 2;
        target.n = (size_t)m - ((size_t)k + 2);
      }
      if (m != -1 && target.n > 0 && !has_ws(target)) {
        Str href = resolve_link(ctx, target);
        buf_adds(out, "<a href=\"");
        esc_append(out, href);
        buf_adds(out, "\">");
        Str inner = {s.p + i + 1, (size_t)k - (i + 1)};
        inline_render(out, inner, ctx);
        buf_adds(out, "</a>");
        i = (size_t)m + 1;
        continue;
      }
      buf_addc(out, '[');
      i++;
      continue;
    }
    esc_append_char(out, c);
    i++;
  }
}

static void render_markdown(MdCtx *ctx, Source *s) {
  Str *lines = NULL;
  int nlines = 0;
  split_lines(ctx->ar, s->body, &lines, &nlines);
  Buf html = {0};
  SMap ids;
  smap_init(&ids, 32);
  TocEnt *toc = NULL;
  int ntoc = 0, captoc = 0;
  Str excerpt = {s->body.p, 0};
  int have_excerpt = 0;
  int i = 0;
  while (i < nlines) {
    Str line = lines[i];
    if (is_blank_line2(line)) {
      i++;
      continue;
    }
    Str lang;
    if (fence_open(line, &lang)) {
      buf_adds(&html, "<pre><code");
      if (lang.n > 0) {
        buf_adds(&html, " class=\"language-");
        esc_append(&html, lang);
        buf_addc(&html, '"');
      }
      buf_addc(&html, '>');
      i++;
      while (i < nlines && !fence_close(lines[i])) {
        esc_append(&html, lines[i]);
        buf_addc(&html, '\n');
        i++;
      }
      i++;
      buf_adds(&html, "</code></pre>\n");
      continue;
    }
    int depth;
    Str htext;
    if (heading_of(line, &depth, &htext)) {
      Str id = unique_slug(ctx->ar, slugify(ctx->ar, htext), &ids);
      Buf ib = {0};
      inline_render(&ib, htext, ctx);
      Str ih = buf_to_str(ctx->ar, &ib);
      buf_free(&ib);
      buf_adds(&html, "<h");
      buf_addc(&html, (unsigned char)('0' + depth));
      buf_adds(&html, " id=\"");
      buf_addn(&html, id.p, id.n);
      buf_adds(&html, "\">");
      buf_addn(&html, ih.p, ih.n);
      buf_adds(&html, "</h");
      buf_addc(&html, (unsigned char)('0' + depth));
      buf_adds(&html, ">\n");
      if (depth == 2 || depth == 3) {
        if (ntoc == captoc) {
          int nc = captoc ? captoc * 2 : 8;
          toc = (TocEnt *)ar_grow_copy(ctx->ar, toc, (size_t)ntoc * sizeof(TocEnt), (size_t)nc * sizeof(TocEnt));
          captoc = nc;
        }
        toc[ntoc].level = depth;
        toc[ntoc].id = id;
        toc[ntoc].html = ih;
        ntoc++;
      }
      i++;
      continue;
    }
    if (is_hr(line)) {
      buf_adds(&html, "<hr>\n");
      i++;
      continue;
    }
    if (line.n > 0 && line.p[0] == '>') {
      Str *ql = NULL;
      int nq = 0, capq = 0;
      while (i < nlines && lines[i].n > 0 && lines[i].p[0] == '>') {
        Str l = lines[i];
        l.p++;
        l.n--;
        if (l.n > 0 && l.p[0] == ' ') {
          l.p++;
          l.n--;
        }
        Str t = trim_line(l);
        if (t.n > 0) {
          if (nq == capq) {
            int nc = capq ? capq * 2 : 8;
            ql = (Str *)ar_grow_copy(ctx->ar, ql, (size_t)nq * sizeof(Str), (size_t)nc * sizeof(Str));
            capq = nc;
          }
          ql[nq++] = t;
        }
        i++;
      }
      if (nq == 0) {
        buf_adds(&html, "<blockquote>\n</blockquote>\n");
      } else {
        buf_adds(&html, "<blockquote>\n<p>");
        Buf jb = {0};
        for (int k = 0; k < nq; k++) {
          if (k) buf_addc(&jb, '\n');
          buf_addn(&jb, ql[k].p, ql[k].n);
        }
        Str joined = {jb.b, jb.len};
        inline_render(&html, joined, ctx);
        buf_free(&jb);
        buf_adds(&html, "</p>\n</blockquote>\n");
      }
      continue;
    }
    {
      Str rest;
      long num = 0;
      int is_bullet = match_bullet(line, &rest);
      int is_ord = 0;
      if (!is_bullet) is_ord = match_ordered(line, &num, &rest);
      if (is_bullet || is_ord) {
        long start = is_ord ? num : 1;
        /* Items, each a Buf joined from its parts. */
        Buf *items = NULL;
        int nitems = 0, capitems = 0;
        while (i < nlines) {
          Str r2;
          long n2 = 0;
          int m2 = is_ord ? match_ordered(lines[i], &n2, &r2) : match_bullet(lines[i], &r2);
          if (m2) {
            if (nitems == capitems) {
              int nc = capitems ? capitems * 2 : 8;
              Buf *ni = (Buf *)malloc((size_t)nc * sizeof(Buf));
              if (!ni) abort();
              if (nitems) memcpy(ni, items, (size_t)nitems * sizeof(Buf));
              free(items);
              items = ni;
              capitems = nc;
            }
            items[nitems].b = NULL;
            items[nitems].len = items[nitems].cap = 0;
            Str t = trim_line(r2);
            buf_addn(&items[nitems], t.p, t.n);
            nitems++;
            i++;
          } else if (nitems > 0 && is_continued(lines[i])) {
            Str t = trim_line(lines[i]);
            buf_addc(&items[nitems - 1], '\n');
            buf_addn(&items[nitems - 1], t.p, t.n);
            i++;
          } else {
            break;
          }
        }
        if (is_ord && start != 1) {
          char tmp[32];
          int m = snprintf(tmp, sizeof(tmp), "<ol start=\"%ld\">\n", start);
          buf_addn(&html, tmp, (size_t)m);
        } else if (is_ord) {
          buf_adds(&html, "<ol>\n");
        } else {
          buf_adds(&html, "<ul>\n");
        }
        for (int k = 0; k < nitems; k++) {
          buf_adds(&html, "<li>");
          Str it = {items[k].b, items[k].len};
          inline_render(&html, it, ctx);
          buf_adds(&html, "</li>\n");
          buf_free(&items[k]);
        }
        free(items);
        buf_adds(&html, is_ord ? "</ol>\n" : "</ul>\n");
        continue;
      }
    }
    /* Paragraph. */
    {
      Buf pb = {0};
      Str t0 = trim_line(line);
      buf_addn(&pb, t0.p, t0.n);
      i++;
      while (i < nlines && !ends_paragraph(lines[i])) {
        Str t = trim_line(lines[i]);
        buf_addc(&pb, '\n');
        buf_addn(&pb, t.p, t.n);
        i++;
      }
      Buf ib = {0};
      Str ptext = {pb.b, pb.len};
      inline_render(&ib, ptext, ctx);
      if (!have_excerpt) {
        excerpt = buf_to_str(ctx->ar, &ib);
        have_excerpt = 1;
      }
      buf_adds(&html, "<p>");
      buf_addn(&html, ib.b, ib.len);
      buf_adds(&html, "</p>\n");
      buf_free(&pb);
      buf_free(&ib);
    }
  }
  s->html = buf_to_str(ctx->ar, &html);
  buf_free(&html);
  s->toc = toc;
  s->ntoc = ntoc;
  s->excerpt = excerpt;
  smap_free(&ids);
}

/* ------------------------------------------------------------ template values */

typedef enum { V_NULL, V_BOOL, V_NUM, V_STR, V_ARR, V_OBJ } VK;

typedef struct {
  Str name;
  Val *v;
} Field;

struct Val {
  VK k;
  long long n;
  int b;
  Str s;
  Val **items;
  int len;
  Field *fields;
  int nfields;
};

static Val VNULL_SINGLETON = {V_NULL, 0, 0, {NULL, 0}, NULL, 0, NULL, 0};

static Val *v_new(Arena *ar, VK k) {
  Val *v = (Val *)ar_alloc(ar, sizeof(Val));
  memset(v, 0, sizeof(*v));
  v->k = k;
  return v;
}

static Val *v_bool(Arena *ar, int b) {
  Val *v = v_new(ar, V_BOOL);
  v->b = b;
  return v;
}

static Val *v_num(Arena *ar, long long n) {
  Val *v = v_new(ar, V_NUM);
  v->n = n;
  return v;
}

static Val *v_str(Arena *ar, Str s) {
  Val *v = v_new(ar, V_STR);
  v->s = s;
  return v;
}

static Val *v_arr(Arena *ar, Val **items, int len) {
  Val *v = v_new(ar, V_ARR);
  v->items = items;
  v->len = len;
  return v;
}

static Val *v_obj(Arena *ar, Field *fields, int nfields) {
  Val *v = v_new(ar, V_OBJ);
  v->fields = fields;
  v->nfields = nfields;
  return v;
}

static void val_text(Buf *out, Val *v) {
  switch (v->k) {
    case V_STR:
      buf_addn(out, v->s.p, v->s.n);
      break;
    case V_NUM: {
      char tmp[32];
      int m = snprintf(tmp, sizeof(tmp), "%lld", v->n);
      buf_addn(out, tmp, (size_t)m);
      break;
    }
    case V_BOOL:
      buf_adds(out, v->b ? "true" : "false");
      break;
    case V_ARR:
      for (int i = 0; i < v->len; i++) {
        if (i) buf_adds(out, ", ");
        val_text(out, v->items[i]);
      }
      break;
    default:
      break;
  }
}

static void val_text_esc(Buf *out, Val *v) {
  switch (v->k) {
    case V_STR:
      esc_append(out, v->s);
      break;
    case V_NUM: {
      char tmp[32];
      int m = snprintf(tmp, sizeof(tmp), "%lld", v->n);
      buf_addn(out, tmp, (size_t)m);
      break;
    }
    case V_BOOL:
      buf_adds(out, v->b ? "true" : "false");
      break;
    case V_ARR:
      for (int i = 0; i < v->len; i++) {
        if (i) buf_adds(out, ", ");
        val_text_esc(out, v->items[i]);
      }
      break;
    default:
      break;
  }
}

static int truthy(Val *v) {
  switch (v->k) {
    case V_ARR: return v->len > 0;
    case V_NULL: return 0;
    case V_BOOL: return v->b;
    case V_STR: return v->s.n > 0;
    case V_NUM: return v->n != 0;
    default: return 1;
  }
}

typedef struct Scope Scope;
struct Scope {
  Scope *parent;
  int is_for;
  Str varname;
  Val *elem;
  Val *loop;
  Field *vars;
  int nvars;
};

static Val *lookup(Arena *ar, Scope *sc, Str *segs, int nseg) {
  Val *v = NULL;
  Scope *s = sc;
  while (s) {
    if (s->is_for) {
      if (str_eq_c(segs[0], "loop")) {
        v = s->loop;
        break;
      }
      if (str_eq(segs[0], s->varname)) {
        v = s->elem;
        break;
      }
    } else {
      for (int i = 0; i < s->nvars; i++) {
        if (str_eq(segs[0], s->vars[i].name)) {
          v = s->vars[i].v;
          break;
        }
      }
      if (v) break;
    }
    s = s->parent;
  }
  if (!v) return NULL;
  for (int k = 1; k < nseg; k++) {
    if (v->k == V_OBJ) {
      Val *nx = NULL;
      for (int i = 0; i < v->nfields; i++) {
        if (str_eq(segs[k], v->fields[i].name)) {
          nx = v->fields[i].v;
          break;
        }
      }
      if (!nx) return NULL;
      v = nx;
    } else if (v->k == V_ARR) {
      if (str_eq_c(segs[k], "length")) {
        v = v_num(ar, v->len);
      } else {
        Str sg = segs[k];
        if (sg.n == 0) return NULL;
        if (sg.n > 1 && sg.p[0] == '0') return NULL;
        unsigned long long acc = 0;
        for (size_t t = 0; t < sg.n; t++) {
          if (sg.p[t] < '0' || sg.p[t] > '9') return NULL;
          acc = acc * 10 + (unsigned long long)(sg.p[t] - '0');
          if (acc >= (unsigned long long)v->len) return NULL;
        }
        v = v->items[acc];
      }
    } else {
      return NULL;
    }
  }
  return v;
}

/* ------------------------------------------------------------ template nodes */

typedef enum { ND_TEXT, ND_VAR, ND_RAW, ND_FOR, ND_IF, ND_INC } NK;

typedef struct Node Node;
struct Node {
  NK k;
  Str text; /* TEXT: literal; INC: template name */
  Str *path;
  int npath;
  Str varname; /* FOR */
  int notflag; /* IF */
  Node **body;
  int nbody, capbody;
  Node **els;
  int nels, capels;
  int has_else;
};

static Node *node_new(Arena *ar, NK k) {
  Node *n = (Node *)ar_alloc(ar, sizeof(Node));
  memset(n, 0, sizeof(*n));
  n->k = k;
  return n;
}

static void node_push(Arena *ar, Node ***arr, int *n, int *cap, Node *nd) {
  if (*n == *cap) {
    int nc = *cap ? *cap * 2 : 4;
    *arr = (Node **)ar_grow_copy(ar, *arr, (size_t)(*n) * sizeof(Node *), (size_t)nc * sizeof(Node *));
    *cap = nc;
  }
  (*arr)[(*n)++] = nd;
}

typedef struct {
  Node *n;
  int use_else;
} Frame;

static void append_cur(Arena *ar, Frame *st, int sp, Node ***root, int *nr, int *cr, Node *nd) {
  if (sp == 1) {
    node_push(ar, root, nr, cr, nd);
    return;
  }
  Node *top = st[sp - 1].n;
  if (top->k == ND_FOR) {
    node_push(ar, &top->body, &top->nbody, &top->capbody, nd);
  } else if (st[sp - 1].use_else) {
    node_push(ar, &top->els, &top->nels, &top->capels, nd);
  } else {
    node_push(ar, &top->body, &top->nbody, &top->capbody, nd);
  }
}

static int parse_path(Arena *ar, Str t, Str **out_segs, int *out_n) {
  if (t.n == 0) return -1;
  int n = 1;
  for (size_t i = 0; i < t.n; i++)
    if (t.p[i] == '.') n++;
  Str *segs = (Str *)ar_alloc(ar, (size_t)n * sizeof(Str));
  int ns = 0;
  size_t start = 0;
  for (size_t i = 0; i <= t.n; i++) {
    if (i == t.n || t.p[i] == '.') {
      Str seg = {t.p + start, i - start};
      if (seg.n == 0) return -1;
      unsigned char c0 = seg.p[0];
      if (!((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z') || c0 == '_')) return -1;
      for (size_t u = 1; u < seg.n; u++) {
        unsigned char c = seg.p[u];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return -1;
      }
      segs[ns++] = seg;
      start = i + 1;
    }
  }
  *out_segs = segs;
  *out_n = ns;
  return 0;
}

static long find_sub(Str s, const char *needle, size_t from) {
  size_t nl = strlen(needle);
  if (nl == 0 || from > s.n) return -1;
  for (size_t i = from; i + nl <= s.n; i++) {
    if (s.p[i] == (unsigned char)needle[0] && memcmp(s.p + i, needle, nl) == 0) return (long)i;
  }
  return -1;
}

static int all_spaces(Str s, size_t a, size_t b) {
  for (size_t i = a; i < b; i++)
    if (s.p[i] != ' ' && s.p[i] != '\t') return 0;
  return 1;
}

static int is_name_char(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

/* Parse a template. Returns 0 ok, -1 with *err set. */
static int parse_template(Arena *ar, Str name, Str src, Node ***out_body, int *out_n, Str *err) {
  Buf text = {0};
  Node **root = NULL;
  int nroot = 0, caproot = 0;
  Frame *st = NULL;
  int sp = 0, capst = 0;
  int rc = -1;

#define TPL_ERR_REASON(_reason_str) \
  do { \
    Buf _b = {0}; \
    buf_adds(&_b, "template \""); \
    buf_addn(&_b, name.p, name.n); \
    buf_adds(&_b, "\": "); \
    buf_addn(&_b, (_reason_str).p, (_reason_str).n); \
    *err = buf_to_str(ar, &_b); \
    buf_free(&_b); \
    goto done; \
  } while (0)

  {
    Frame f0 = {NULL, 0};
    st = (Frame *)malloc(8 * sizeof(Frame));
    if (!st) abort();
    capst = 8;
    st[sp++] = f0;
  }

  size_t i = 0;
  while (i < src.n) {
    const unsigned char *op = (const unsigned char *)memchr(src.p + i, '{', src.n - i);
    if (!op) {
      buf_addn(&text, src.p + i, src.n - i);
      break;
    }
    size_t open = (size_t)(op - src.p);
    if (open == src.n - 1) {
      buf_addn(&text, src.p + i, src.n - i);
      break;
    }
    unsigned char nx = src.p[open + 1];
    if (nx != '{' && nx != '%') {
      buf_addn(&text, src.p + i, open + 1 - i);
      i = open + 1;
      continue;
    }
    buf_addn(&text, src.p + i, open - i);
    if (nx == '{') {
      int raw = (open + 2 < src.n && src.p[open + 2] == '{');
      const char *needle = raw ? "}}}" : "}}";
      long close = find_sub(src, needle, open + (raw ? 3 : 2));
      if (close < 0) {
        Str reason = str_from_c(raw ? "a {{{ is not closed" : "a {{ is not closed");
        TPL_ERR_REASON(reason);
      }
      if (text.len > 0) {
        Node *nd = node_new(ar, ND_TEXT);
        nd->text = buf_to_str(ar, &text);
        text.len = 0;
        append_cur(ar, st, sp, &root, &nroot, &caproot, nd);
      }
      Str inner = {src.p + open + (raw ? 3 : 2), (size_t)close - (open + (raw ? 3 : 2))};
      Str trimmed = trim_str(inner);
      Str *segs = NULL;
      int nseg = 0;
      if (parse_path(ar, trimmed, &segs, &nseg) != 0) {
        Buf rb = {0};
        json_quote(&rb, trimmed);
        buf_adds(&rb, " is not a path");
        Str reason = buf_to_str(ar, &rb);
        buf_free(&rb);
        TPL_ERR_REASON(reason);
      }
      Node *nd = node_new(ar, raw ? ND_RAW : ND_VAR);
      nd->path = segs;
      nd->npath = nseg;
      append_cur(ar, st, sp, &root, &nroot, &caproot, nd);
      i = (size_t)close + (raw ? 3 : 2);
      continue;
    }
    /* {% %} */
    long close = find_sub(src, "%}", open + 2);
    if (close < 0) {
      Str reason = str_from_c("a {% is not closed");
      TPL_ERR_REASON(reason);
    }
    size_t end = (size_t)close + 2;
    size_t ls = 0;
    for (size_t t = open; t > 0; t--) {
      if (src.p[t - 1] == '\n') {
        ls = t;
        break;
      }
    }
    const unsigned char *nlp = (const unsigned char *)memchr(src.p + end, '\n', src.n - end);
    size_t nl = nlp ? (size_t)(nlp - src.p) : (size_t)-1;
    size_t lineEnd = nl == (size_t)-1 ? src.n : nl;
    int standalone = all_spaces(src, ls, open) && all_spaces(src, end, lineEnd);
    if (standalone) {
      size_t drop = open - ls;
      text.len = drop > text.len ? 0 : text.len - drop;
    }
    if (text.len > 0) {
      Node *nd = node_new(ar, ND_TEXT);
      nd->text = buf_to_str(ar, &text);
      text.len = 0;
      append_cur(ar, st, sp, &root, &nroot, &caproot, nd);
    }
    Str tag = trim_str((Str){src.p + open + 2, (size_t)close - (open + 2)});
    /* Dispatch the tag. */
    if (tag.n >= 4 && memcmp(tag.p, "for ", 4) == 0) {
      Str rest = {tag.p + 4, tag.n - 4};
      size_t k = 0;
      if (rest.n > 0 && ((rest.p[0] >= 'A' && rest.p[0] <= 'Z') || (rest.p[0] >= 'a' && rest.p[0] <= 'z') || rest.p[0] == '_')) {
        k = 1;
        while (k < rest.n && is_name_char(rest.p[k])) k++;
      }
      int ok = 0;
      Str vname = {0, 0}, pstr = {0, 0};
      if (k > 0 && k + 4 <= rest.n && memcmp(rest.p + k, " in ", 4) == 0) {
        vname.p = rest.p;
        vname.n = k;
        pstr.p = rest.p + k + 4;
        pstr.n = rest.n - (k + 4);
        if (pstr.n > 0 && !has_ws(pstr)) ok = 1;
      }
      if (!ok) goto not_a_tag;
      Str *segs = NULL;
      int nseg = 0;
      if (parse_path(ar, pstr, &segs, &nseg) != 0) {
        Buf rb = {0};
        json_quote(&rb, pstr);
        buf_adds(&rb, " is not a path");
        Str reason = buf_to_str(ar, &rb);
        buf_free(&rb);
        TPL_ERR_REASON(reason);
      }
      Node *nd = node_new(ar, ND_FOR);
      nd->varname = vname;
      nd->path = segs;
      nd->npath = nseg;
      append_cur(ar, st, sp, &root, &nroot, &caproot, nd);
      if (sp == capst) {
        int nc = capst * 2;
        Frame *ns = (Frame *)realloc(st, (size_t)nc * sizeof(Frame));
        if (!ns) abort();
        st = ns;
        capst = nc;
      }
      st[sp].n = nd;
      st[sp].use_else = 0;
      sp++;
    } else if (tag.n >= 3 && memcmp(tag.p, "if ", 3) == 0) {
      Str rest = {tag.p + 3, tag.n - 3};
      int notflag = 0;
      if (rest.n >= 4 && memcmp(rest.p, "not ", 4) == 0) {
        notflag = 1;
        rest.p += 4;
        rest.n -= 4;
      }
      if (rest.n == 0 || has_ws(rest)) goto not_a_tag;
      Str *segs = NULL;
      int nseg = 0;
      if (parse_path(ar, rest, &segs, &nseg) != 0) {
        Buf rb = {0};
        json_quote(&rb, rest);
        buf_adds(&rb, " is not a path");
        Str reason = buf_to_str(ar, &rb);
        buf_free(&rb);
        TPL_ERR_REASON(reason);
      }
      Node *nd = node_new(ar, ND_IF);
      nd->notflag = notflag;
      nd->path = segs;
      nd->npath = nseg;
      append_cur(ar, st, sp, &root, &nroot, &caproot, nd);
      if (sp == capst) {
        int nc = capst * 2;
        Frame *ns = (Frame *)realloc(st, (size_t)nc * sizeof(Frame));
        if (!ns) abort();
        st = ns;
        capst = nc;
      }
      st[sp].n = nd;
      st[sp].use_else = 0;
      sp++;
    } else if (str_eq_c(tag, "else")) {
      if (sp <= 1 || st[sp - 1].n->k != ND_IF || st[sp - 1].use_else) {
        Str reason = str_from_c("an else outside an if");
        TPL_ERR_REASON(reason);
      }
      st[sp - 1].use_else = 1;
      st[sp - 1].n->has_else = 1;
    } else if (str_eq_c(tag, "endfor") || str_eq_c(tag, "endif")) {
      NK want = str_eq_c(tag, "endfor") ? ND_FOR : ND_IF;
      if (sp <= 1 || st[sp - 1].n->k != want) {
        Str reason = str_from_c(str_eq_c(tag, "endfor") ? "an endfor with no open for" : "an endif with no open if");
        TPL_ERR_REASON(reason);
      }
      sp--;
    } else if (tag.n >= 9 && memcmp(tag.p, "include ", 8) == 0) {
      /* /^include "([a-z0-9-]+)"$/ */
      int ok = 0;
      Str nm = {0, 0};
      if (tag.n >= 11 && tag.p[8] == '"' && tag.p[tag.n - 1] == '"') {
        nm.p = tag.p + 9;
        nm.n = tag.n - 10;
        ok = nm.n > 0;
        for (size_t t = 0; ok && t < nm.n; t++) {
          unsigned char ch = nm.p[t];
          if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-')) ok = 0;
        }
      }
      if (!ok) goto not_a_tag;
      Node *nd = node_new(ar, ND_INC);
      nd->text = nm;
      append_cur(ar, st, sp, &root, &nroot, &caproot, nd);
    } else {
    not_a_tag:;
      Buf rb = {0};
      buf_adds(&rb, "{% ");
      buf_addn(&rb, tag.p, tag.n);
      buf_adds(&rb, " %} is not a tag");
      Str reason = buf_to_str(ar, &rb);
      buf_free(&rb);
      TPL_ERR_REASON(reason);
    }
    i = standalone ? (nl == (size_t)-1 ? src.n : nl + 1) : end;
  }
  if (text.len > 0) {
    Node *nd = node_new(ar, ND_TEXT);
    nd->text = buf_to_str(ar, &text);
    text.len = 0;
    append_cur(ar, st, sp, &root, &nroot, &caproot, nd);
  }
  if (sp > 1) {
    Node *top = st[sp - 1].n;
    Str reason = str_from_c(top->k == ND_FOR ? "a for is not closed" : "a if is not closed");
    TPL_ERR_REASON(reason);
  }
  *out_body = root;
  *out_n = nroot;
  rc = 0;

done:
  buf_free(&text);
  free(st);
  return rc;
#undef TPL_ERR_REASON
}

/* ------------------------------------------------------------ template render */

typedef struct {
  Str name;
  Node **body;
  int nbody;
} PTmpl;

typedef struct {
  SMap *tmap; /* name -> index into ptmpls */
  PTmpl *ptmpls;
  Arena *ar;
} RCtx;

static void render_nodes(Buf *out, Node **nodes, int n, Scope *sc, RCtx *ctx, int depth) {
  for (int i = 0; i < n; i++) {
    Node *nd = nodes[i];
    switch (nd->k) {
      case ND_TEXT:
        buf_addn(out, nd->text.p, nd->text.n);
        break;
      case ND_VAR:
      case ND_RAW: {
        Val *v = lookup(ctx->ar, sc, nd->path, nd->npath);
        if (v) {
          if (nd->k == ND_VAR) val_text_esc(out, v);
          else val_text(out, v);
        }
        break;
      }
      case ND_IF: {
        Val *v = lookup(ctx->ar, sc, nd->path, nd->npath);
        int tv = v ? truthy(v) : 0;
        Node **br = NULL;
        int nbr = 0;
        if (tv != nd->notflag) {
          br = nd->body;
          nbr = nd->nbody;
        } else if (nd->has_else) {
          br = nd->els;
          nbr = nd->nels;
        }
        if (br) render_nodes(out, br, nbr, sc, ctx, depth);
        break;
      }
      case ND_FOR: {
        Val *v = lookup(ctx->ar, sc, nd->path, nd->npath);
        if (!v || v->k != V_ARR) break;
        Arena *ar = ctx->ar;
        Val *vi = v_num(ar, 0), *vf = v_bool(ar, 0), *vl = v_bool(ar, 0), *vn = v_num(ar, v->len);
        Field *lf = (Field *)ar_alloc(ar, 4 * sizeof(Field));
        lf[0].name = str_from_c("index");
        lf[0].v = vi;
        lf[1].name = str_from_c("first");
        lf[1].v = vf;
        lf[2].name = str_from_c("last");
        lf[2].v = vl;
        lf[3].name = str_from_c("length");
        lf[3].v = vn;
        Val *loop = v_obj(ar, lf, 4);
        Scope sc2;
        sc2.parent = sc;
        sc2.is_for = 1;
        sc2.varname = nd->varname;
        sc2.elem = NULL;
        sc2.loop = loop;
        sc2.vars = NULL;
        sc2.nvars = 0;
        for (int k = 0; k < v->len; k++) {
          vi->n = k + 1;
          vf->b = k == 0;
          vl->b = k == v->len - 1;
          sc2.elem = v->items[k];
          render_nodes(out, nd->body, nd->nbody, &sc2, ctx, depth);
        }
        break;
      }
      case ND_INC: {
        int idx = smap_get(ctx->tmap, nd->text);
        if (idx >= 0 && depth < 8)
          render_nodes(out, ctx->ptmpls[idx].body, ctx->ptmpls[idx].nbody, sc, ctx, depth + 1);
        break;
      }
    }
  }
}

/* ------------------------------------------------------------- pipeline */

typedef struct {
  Str path, text;
} OutEnt;

typedef struct {
  Arena *ar;
  CaseIn *in;
  Str base;
  PTmpl *ptmpls;
  int nptmpl;
  SMap tmap;
  Source **srcs;
  int nsrcs;
  SMap by_src;
  SMap statics;
  FileEnt **statics_list;
  int nstatics;
  Str *report;
  int nreport, capreport;
  Source **posts;
  int nposts;
  Tag **tags;
  int ntags;
  MdCtx mctx;
  OutEnt *out;
  int nout, capout;
} Build;

static void report_push(Build *b, Str line) {
  if (b->nreport == b->capreport) {
    int nc = b->capreport ? b->capreport * 2 : 32;
    b->report = (Str *)ar_grow_copy(b->ar, b->report, (size_t)b->nreport * sizeof(Str), (size_t)nc * sizeof(Str));
    b->capreport = nc;
  }
  b->report[b->nreport++] = line;
}

static void out_push(Build *b, Str path, Str text) {
  if (b->nout == b->capout) {
    int nc = b->capout ? b->capout * 2 : 64;
    b->out = (OutEnt *)ar_grow_copy(b->ar, b->out, (size_t)b->nout * sizeof(OutEnt), (size_t)nc * sizeof(OutEnt));
    b->capout = nc;
  }
  b->out[b->nout].path = path;
  b->out[b->nout].text = text;
  b->nout++;
}

static int out_has(Build *b, Str path) {
  for (int i = 0; i < b->nout; i++)
    if (str_eq(b->out[i].path, path)) return 1;
  return 0;
}

static Str dir_of(Str path) {
  for (size_t i = path.n; i > 0; i--) {
    if (path.p[i - 1] == '/') {
      Str r = {path.p, i - 1};
      return r;
    }
  }
  Str r = {path.p, 0};
  return r;
}

static Str normalize_crlf(Arena *ar, Str s) {
  /* Fast path: no CR at all means the text is already normalised. */
  if (!memchr(s.p, '\r', s.n)) return s;
  unsigned char *d = (unsigned char *)ar_alloc(ar, s.n ? s.n : 1);
  size_t w = 0;
  for (size_t i = 0; i < s.n; i++) {
    if (s.p[i] == '\r') {
      d[w++] = '\n';
      if (i + 1 < s.n && s.p[i + 1] == '\n') i++;
    } else {
      d[w++] = s.p[i];
    }
  }
  Str r = {d, w};
  return r;
}

static const char *MONTHS[] = {"January", "February", "March", "April", "May", "June", "July",
                               "August", "September", "October", "November", "December"};

static Str long_date(Arena *ar, Str date) {
  int d = (date.p[8] - '0') * 10 + (date.p[9] - '0');
  int m = (date.p[5] - '0') * 10 + (date.p[6] - '0');
  Buf b = {0};
  char tmp[16];
  int n = snprintf(tmp, sizeof(tmp), "%d %s ", d, MONTHS[m - 1]);
  buf_addn(&b, tmp, (size_t)n);
  buf_addn(&b, date.p, 4);
  Str r = buf_to_str(ar, &b);
  buf_free(&b);
  return r;
}

static int cmp_fileent_ptr(const void *a, const void *b) {
  FileEnt *x = *(FileEnt *const *)a, *y = *(FileEnt *const *)b;
  return str_cmp_u16(x->path, y->path);
}

static int cmp_tmpl_ptr(const void *a, const void *b) {
  Tmpl *x = *(Tmpl *const *)a, *y = *(Tmpl *const *)b;
  return str_cmp_u16(x->name, y->name);
}

static int cmp_str_el(const void *a, const void *b) {
  return str_cmp_u16(*(const Str *)a, *(const Str *)b);
}

static int cmp_post_ptr(const void *a, const void *b) {
  Source *x = *(Source *const *)a, *y = *(Source *const *)b;
  int c = str_cmp_u16(y->date, x->date);
  if (c) return c;
  return str_cmp_u16(x->slugf, y->slugf);
}

static int cmp_tag_ptr(const void *a, const void *b) {
  Tag *x = *(Tag *const *)a, *y = *(Tag *const *)b;
  if (x->nposts != y->nposts) return y->nposts - x->nposts;
  return str_cmp_u16(x->slug, y->slug);
}

typedef struct {
  Str dir;
  SMap set;
} TakenDir;

static SMap *taken_for(Arena *ar, TakenDir **tds, int *n, int *cap, Str dir) {
  for (int i = 0; i < *n; i++)
    if (str_eq((*tds)[i].dir, dir)) return &(*tds)[i].set;
  if (*n == *cap) {
    int nc = *cap ? *cap * 2 : 8;
    *tds = (TakenDir *)ar_grow_copy(ar, *tds, (size_t)(*n) * sizeof(TakenDir), (size_t)nc * sizeof(TakenDir));
    *cap = nc;
  }
  TakenDir *t = &(*tds)[(*n)++];
  t->dir = dir;
  smap_init(&t->set, 16);
  if (dir.n == 0) {
    smap_put(&t->set, str_from_c("posts"), 1);
    smap_put(&t->set, str_from_c("tags"), 1);
    smap_put(&t->set, str_from_c("page"), 1);
  }
  return &t->set;
}

static void card_fields(Build *b, Source *s, Field **out_f, int *out_n) {
  Arena *ar = b->ar;
  Field *f = (Field *)ar_alloc(ar, 8 * sizeof(Field));
  Str empty = str_from_c("");
  f[0].name = str_from_c("title");
  f[0].v = v_str(ar, s->title);
  f[1].name = str_from_c("url");
  f[1].v = v_str(ar, s->url);
  f[2].name = str_from_c("date");
  f[2].v = v_str(ar, s->has_date ? s->date : empty);
  f[3].name = str_from_c("date_long");
  f[3].v = v_str(ar, s->date_long);
  f[4].name = str_from_c("summary");
  f[4].v = v_str(ar, s->has_summary ? s->summary : empty);
  f[5].name = str_from_c("excerpt");
  f[5].v = v_str(ar, s->excerpt);
  f[6].name = str_from_c("reading_minutes");
  f[6].v = v_num(ar, s->read_min);
  Val **titems = NULL;
  if (s->ntagrefs > 0) titems = (Val **)ar_alloc(ar, (size_t)s->ntagrefs * sizeof(Val *));
  for (int k = 0; k < s->ntagrefs; k++) {
    Tag *t = s->tagrefs[k];
    Field *tf = (Field *)ar_alloc(ar, 3 * sizeof(Field));
    tf[0].name = str_from_c("name");
    tf[0].v = v_str(ar, t->name);
    tf[1].name = str_from_c("slug");
    tf[1].v = v_str(ar, t->slug);
    tf[2].name = str_from_c("url");
    tf[2].v = v_str(ar, t->url);
    titems[k] = v_obj(ar, tf, 3);
  }
  f[7].name = str_from_c("tags");
  f[7].v = v_arr(ar, titems, s->ntagrefs);
  *out_f = f;
  *out_n = 8;
}

static Val *link_val(Build *b, Source *s) {
  Arena *ar = b->ar;
  Str empty = str_from_c("");
  Field *f = (Field *)ar_alloc(ar, 4 * sizeof(Field));
  f[0].name = str_from_c("title");
  f[0].v = v_str(ar, s->title);
  f[1].name = str_from_c("url");
  f[1].v = v_str(ar, s->url);
  f[2].name = str_from_c("date");
  f[2].v = v_str(ar, s->has_date ? s->date : empty);
  f[3].name = str_from_c("date_long");
  f[3].v = v_str(ar, s->date_long);
  return v_obj(ar, f, 4);
}

static Val *page_val(Build *b, Source *s) {
  Arena *ar = b->ar;
  Field *cf = NULL;
  int ncf = 0;
  card_fields(b, s, &cf, &ncf);
  Field *f = (Field *)ar_alloc(ar, 14 * sizeof(Field));
  memcpy(f, cf, (size_t)ncf * sizeof(Field));
  int n = ncf;
  f[n].name = str_from_c("slug");
  f[n].v = v_str(ar, s->slugf);
  n++;
  f[n].name = str_from_c("content");
  f[n].v = v_str(ar, s->html);
  n++;
  Val **tocitems = NULL;
  if (s->ntoc > 0) tocitems = (Val **)ar_alloc(ar, (size_t)s->ntoc * sizeof(Val *));
  for (int k = 0; k < s->ntoc; k++) {
    Field *tf = (Field *)ar_alloc(ar, 3 * sizeof(Field));
    tf[0].name = str_from_c("level");
    tf[0].v = v_num(ar, s->toc[k].level);
    tf[1].name = str_from_c("id");
    tf[1].v = v_str(ar, s->toc[k].id);
    tf[2].name = str_from_c("html");
    tf[2].v = v_str(ar, s->toc[k].html);
    tocitems[k] = v_obj(ar, tf, 3);
  }
  f[n].name = str_from_c("toc");
  f[n].v = v_arr(ar, tocitems, s->ntoc);
  n++;
  f[n].name = str_from_c("newer");
  f[n].v = s->newer ? link_val(b, s->newer) : &VNULL_SINGLETON;
  n++;
  f[n].name = str_from_c("older");
  f[n].v = s->older ? link_val(b, s->older) : &VNULL_SINGLETON;
  n++;
  f[n].name = str_from_c("is_post");
  f[n].v = v_bool(ar, s->post);
  n++;
  return v_obj(ar, f, n);
}

static Val *tag_val(Build *b, Tag *t) {
  Arena *ar = b->ar;
  Field *f = (Field *)ar_alloc(ar, 5 * sizeof(Field));
  f[0].name = str_from_c("name");
  f[0].v = v_str(ar, t->name);
  f[1].name = str_from_c("slug");
  f[1].v = v_str(ar, t->slug);
  f[2].name = str_from_c("url");
  f[2].v = v_str(ar, t->url);
  f[3].name = str_from_c("count");
  f[3].v = v_num(ar, t->nposts);
  Val **pitems = NULL;
  if (t->nposts > 0) pitems = (Val **)ar_alloc(ar, (size_t)t->nposts * sizeof(Val *));
  for (int k = 0; k < t->nposts; k++) pitems[k] = t->posts[k]->card;
  f[4].name = str_from_c("posts");
  f[4].v = v_arr(ar, pitems, t->nposts);
  return v_obj(ar, f, 5);
}

static Str index_url(Arena *ar, Str base, long long k) {
  Buf b = {0};
  buf_addn(&b, base.p, base.n);
  if (k == 1) {
    buf_addc(&b, '/');
  } else {
    char tmp[48];
    int m = snprintf(tmp, sizeof(tmp), "/page/%lld/", k);
    buf_addn(&b, tmp, (size_t)m);
  }
  Str r = buf_to_str(ar, &b);
  buf_free(&b);
  return r;
}

static int render_named(Build *b, Str name, Scope *sc, Str *out) {
  int idx = smap_get(&b->tmap, name);
  if (idx < 0) return 0;
  Buf buf = {0};
  RCtx ctx = {&b->tmap, b->ptmpls, b->ar};
  render_nodes(&buf, b->ptmpls[idx].body, b->ptmpls[idx].nbody, sc, &ctx, 0);
  *out = buf_to_str(b->ar, &buf);
  buf_free(&buf);
  return 1;
}

/* Build one case's answer JSON into out (a Buf). */
static void build_case(Arena *ar, CaseIn *in, Buf *out_json) {
  Build b;
  memset(&b, 0, sizeof(b));
  b.ar = ar;
  b.in = in;

  /* 1. Sources, sorted by path; .md normalised, the rest static. */
  FileEnt **files = NULL;
  if (in->nfile > 0) {
    files = (FileEnt **)ar_alloc(ar, (size_t)in->nfile * sizeof(FileEnt *));
    for (int i = 0; i < in->nfile; i++) files[i] = &in->files[i];
    void *tmp = ar_alloc(ar, (size_t)in->nfile * sizeof(FileEnt *));
    msort_gen(files, (size_t)in->nfile, sizeof(FileEnt *), tmp, cmp_fileent_ptr);
  }
  Source **all_src = NULL;
  int nall = 0, capall = 0;
  FileEnt **statics_list = NULL;
  int nstatics = 0, capstatics = 0;
  smap_init(&b.statics, 64);
  for (int i = 0; i < in->nfile; i++) {
    FileEnt *f = files[i];
    if (str_ends(f->path, ".md")) {
      Source *s = (Source *)ar_alloc(ar, sizeof(Source));
      memset(s, 0, sizeof(*s));
      s->path = f->path;
      s->text = normalize_crlf(ar, f->content);
      s->post = str_starts(f->path, "posts/");
      if (nall == capall) {
        int nc = capall ? capall * 2 : 64;
        all_src = (Source **)ar_grow_copy(ar, all_src, (size_t)nall * sizeof(Source *), (size_t)nc * sizeof(Source *));
        capall = nc;
      }
      all_src[nall++] = s;
    } else {
      if (nstatics == capstatics) {
        int nc = capstatics ? capstatics * 2 : 16;
        statics_list =
            (FileEnt **)ar_grow_copy(ar, statics_list, (size_t)nstatics * sizeof(FileEnt *), (size_t)nc * sizeof(FileEnt *));
        capstatics = nc;
      }
      statics_list[nstatics++] = f;
      smap_put(&b.statics, f->path, 1);
    }
  }
  b.statics_list = statics_list;
  b.nstatics = nstatics;

  /* Base URL without trailing slashes. */
  b.base = in->site.base;
  while (b.base.n > 0 && b.base.p[b.base.n - 1] == '/') b.base.n--;

  /* 2. Templates, parsed in code-unit order of name. */
  Tmpl **ts = NULL;
  if (in->ntmpl > 0) {
    ts = (Tmpl **)ar_alloc(ar, (size_t)in->ntmpl * sizeof(Tmpl *));
    for (int i = 0; i < in->ntmpl; i++) ts[i] = &in->tmpls[i];
    void *tmp = ar_alloc(ar, (size_t)in->ntmpl * sizeof(Tmpl *));
    msort_gen(ts, (size_t)in->ntmpl, sizeof(Tmpl *), tmp, cmp_tmpl_ptr);
  }
  smap_init(&b.tmap, 64);
  b.ptmpls = (PTmpl *)ar_alloc(ar, (size_t)(in->ntmpl > 0 ? in->ntmpl : 1) * sizeof(PTmpl));
  for (int i = 0; i < in->ntmpl; i++) {
    Node **body = NULL;
    int nbody = 0;
    Str err = {0, 0};
    if (parse_template(ar, ts[i]->name, ts[i]->src, &body, &nbody, &err) != 0) {
      report_push(&b, err);
      continue;
    }
    b.ptmpls[b.nptmpl].name = ts[i]->name;
    b.ptmpls[b.nptmpl].body = body;
    b.ptmpls[b.nptmpl].nbody = nbody;
    smap_put(&b.tmap, ts[i]->name, b.nptmpl);
    b.nptmpl++;
  }

  /* 3. Front matter, in source order. */
  Source **kept = NULL;
  int nkept = 0, capkept = 0;
  for (int i = 0; i < nall; i++) {
    Source *s = all_src[i];
    FM fm;
    Str ferr = {0, 0};
    if (parse_fm(ar, s->text, &fm, &ferr) != 0) {
      Buf lb = {0};
      buf_adds(&lb, "refused ");
      buf_addn(&lb, s->path.p, s->path.n);
      buf_adds(&lb, ": ");
      buf_addn(&lb, ferr.p, ferr.n);
      report_push(&b, buf_to_str(ar, &lb));
      buf_free(&lb);
      continue;
    }
    s->title = fm.title;
    s->date = fm.date;
    s->has_date = fm.has_date;
    s->tags = fm.tags;
    s->ntags = fm.ntags;
    s->draft = fm.draft;
    s->body = fm.body;
    s->has_layout = fm.has_layout;
    s->layout = fm.layout;
    s->has_slug = fm.has_slug;
    s->slug = fm.slug;
    s->has_summary = fm.has_summary;
    s->summary = fm.summary;
    if (str_starts(s->path, "tags/") || str_starts(s->path, "page/")) {
      const char *d = str_starts(s->path, "tags/") ? "tags" : "page";
      Buf lb = {0};
      buf_adds(&lb, "refused ");
      buf_addn(&lb, s->path.p, s->path.n);
      buf_adds(&lb, ": ");
      buf_adds(&lb, d);
      buf_adds(&lb, "/ is the generator's own");
      report_push(&b, buf_to_str(ar, &lb));
      buf_free(&lb);
      continue;
    }
    if (s->post && !s->has_date) {
      Buf lb = {0};
      buf_adds(&lb, "refused ");
      buf_addn(&lb, s->path.p, s->path.n);
      buf_adds(&lb, ": a post needs a date");
      report_push(&b, buf_to_str(ar, &lb));
      buf_free(&lb);
      continue;
    }
    Str layout = s->has_layout ? s->layout : str_from_c(s->post ? "post" : "page");
    if (!smap_has(&b.tmap, layout)) {
      Buf lb = {0};
      buf_adds(&lb, "refused ");
      buf_addn(&lb, s->path.p, s->path.n);
      buf_adds(&lb, ": layout \"");
      buf_addn(&lb, layout.p, layout.n);
      buf_adds(&lb, "\" is no template that parses");
      report_push(&b, buf_to_str(ar, &lb));
      buf_free(&lb);
      continue;
    }
    s->layout = layout;
    if (s->draft) {
      Buf lb = {0};
      buf_adds(&lb, "draft ");
      buf_addn(&lb, s->path.p, s->path.n);
      report_push(&b, buf_to_str(ar, &lb));
      buf_free(&lb);
      continue;
    }
    if (nkept == capkept) {
      int nc = capkept ? capkept * 2 : 64;
      kept = (Source **)ar_grow_copy(ar, kept, (size_t)nkept * sizeof(Source *), (size_t)nc * sizeof(Source *));
      capkept = nc;
    }
    kept[nkept++] = s;
  }
  b.srcs = kept;
  b.nsrcs = nkept;

  /* 4. Permalinks. */
  TakenDir *tds = NULL;
  int ntds = 0, captds = 0;
  smap_init(&b.by_src, 128);
  for (int i = 0; i < nkept; i++) {
    Source *s = kept[i];
    Str dir = s->post ? str_from_c("posts") : dir_of(s->path);
    SMap *taken = taken_for(ar, &tds, &ntds, &captds, dir);
    Str base_slug = s->has_slug ? s->slug : slugify(ar, s->title);
    s->slugf = unique_slug(ar, base_slug, taken);
    Buf ob = {0};
    if (dir.n > 0) {
      buf_addn(&ob, dir.p, dir.n);
      buf_addc(&ob, '/');
    }
    buf_addn(&ob, s->slugf.p, s->slugf.n);
    s->outdir = buf_to_str(ar, &ob);
    buf_free(&ob);
    Buf ub = {0};
    buf_addn(&ub, b.base.p, b.base.n);
    buf_addc(&ub, '/');
    buf_addn(&ub, s->outdir.p, s->outdir.n);
    buf_addc(&ub, '/');
    s->url = buf_to_str(ar, &ub);
    buf_free(&ub);
    smap_put(&b.by_src, s->path, i);
  }

  /* 5. Markdown. */
  MdCtx *mc = &b.mctx;
  memset(mc, 0, sizeof(*mc));
  mc->ar = ar;
  mc->srcs = kept;
  mc->by_src = &b.by_src;
  mc->statics = &b.statics;
  mc->base = b.base;
  smap_init(mc->broken_set = (SMap *)ar_alloc(ar, sizeof(SMap)), 64);
  for (int i = 0; i < nkept; i++) {
    Source *s = kept[i];
    mc->cur_path = s->path;
    mc->cur_dir = dir_of(s->path);
    render_markdown(mc, s);
    long long w = words_of(s->body);
    long long rm = (w + 199) / 200;
    s->read_min = rm < 1 ? 1 : rm;
    s->date_long = s->has_date ? long_date(ar, s->date) : str_from_c("");
    s->newer = s->older = NULL;
  }

  /* 6. Collections: newest date first, then slug. */
  Source **posts = NULL;
  int nposts = 0;
  if (nkept > 0) {
    posts = (Source **)ar_alloc(ar, (size_t)nkept * sizeof(Source *));
    for (int i = 0; i < nkept; i++)
      if (kept[i]->post) posts[nposts++] = kept[i];
    if (nposts > 1) {
      void *tmp = ar_alloc(ar, (size_t)nposts * sizeof(Source *));
      msort_gen(posts, (size_t)nposts, sizeof(Source *), tmp, cmp_post_ptr);
    }
  }
  b.posts = posts;
  b.nposts = nposts;
  for (int k = 0; k < nposts; k++) {
    posts[k]->newer = k == 0 ? NULL : posts[k - 1];
    posts[k]->older = k == nposts - 1 ? NULL : posts[k + 1];
  }

  /* 7. Tags. */
  SMap by_slug;
  smap_init(&by_slug, 64);
  Tag **tags = NULL;
  int ntags = 0, captags = 0;
  for (int k = 0; k < nposts; k++) {
    Source *s = posts[k];
    /* Its tags, each slug once, in its own order. */
    Str *seen_slugs = NULL;
    int nseen = 0;
    Tag **refs = NULL;
    int nrefs = 0;
    if (s->ntags > 0) {
      seen_slugs = (Str *)ar_alloc(ar, (size_t)s->ntags * sizeof(Str));
      refs = (Tag **)ar_alloc(ar, (size_t)s->ntags * sizeof(Tag *));
    }
    for (int t = 0; t < s->ntags; t++) {
      Str slug = slugify(ar, s->tags[t]);
      int dup = 0;
      for (int u = 0; u < nseen; u++)
        if (str_eq(seen_slugs[u], slug)) dup = 1;
      if (dup) continue;
      seen_slugs[nseen++] = slug;
      int idx = smap_get(&by_slug, slug);
      Tag *tg;
      if (idx < 0) {
        tg = (Tag *)ar_alloc(ar, sizeof(Tag));
        memset(tg, 0, sizeof(*tg));
        tg->name = s->tags[t];
        tg->slug = slug;
        Buf ub = {0};
        buf_addn(&ub, b.base.p, b.base.n);
        buf_adds(&ub, "/tags/");
        buf_addn(&ub, slug.p, slug.n);
        buf_addc(&ub, '/');
        tg->url = buf_to_str(ar, &ub);
        buf_free(&ub);
        if (ntags == captags) {
          int nc = captags ? captags * 2 : 16;
          tags = (Tag **)ar_grow_copy(ar, tags, (size_t)ntags * sizeof(Tag *), (size_t)nc * sizeof(Tag *));
          captags = nc;
        }
        tags[ntags] = tg;
        smap_put(&by_slug, slug, ntags);
        ntags++;
      } else {
        tg = tags[idx];
      }
      if (tg->nposts == tg->cap) {
        int nc = tg->cap ? tg->cap * 2 : 8;
        tg->posts = (Source **)ar_grow_copy(ar, tg->posts, (size_t)tg->nposts * sizeof(Source *), (size_t)nc * sizeof(Source *));
        tg->cap = nc;
      }
      tg->posts[tg->nposts++] = s;
      refs[nrefs++] = tg;
    }
    s->tagrefs = refs;
    s->ntagrefs = nrefs;
  }
  for (int i = 0; i < nkept; i++) {
    Source *s = kept[i];
    if (s->post) continue;
    Str *seen_slugs = NULL;
    int nseen = 0;
    Tag **refs = NULL;
    int nrefs = 0;
    if (s->ntags > 0) {
      seen_slugs = (Str *)ar_alloc(ar, (size_t)s->ntags * sizeof(Str));
      refs = (Tag **)ar_alloc(ar, (size_t)s->ntags * sizeof(Tag *));
    }
    for (int t = 0; t < s->ntags; t++) {
      Str slug = slugify(ar, s->tags[t]);
      int dup = 0;
      for (int u = 0; u < nseen; u++)
        if (str_eq(seen_slugs[u], slug)) dup = 1;
      if (dup) continue;
      seen_slugs[nseen++] = slug;
      int idx = smap_get(&by_slug, slug);
      Tag *tg;
      if (idx < 0) {
        tg = (Tag *)ar_alloc(ar, sizeof(Tag));
        memset(tg, 0, sizeof(*tg));
        tg->name = s->tags[t];
        tg->slug = slug;
        tg->url = str_from_c("");
      } else {
        tg = tags[idx];
      }
      refs[nrefs++] = tg;
    }
    s->tagrefs = refs;
    s->ntagrefs = nrefs;
  }
  if (ntags > 1) {
    void *tmp = ar_alloc(ar, (size_t)ntags * sizeof(Tag *));
    msort_gen(tags, (size_t)ntags, sizeof(Tag *), tmp, cmp_tag_ptr);
  }
  b.tags = tags;
  b.ntags = ntags;

  /* 8. Layouts. */
  Arena *ar2 = ar;
  Val *site_v;
  {
    Field *f = (Field *)ar_alloc(ar2, 2 * sizeof(Field));
    f[0].name = str_from_c("title");
    f[0].v = v_str(ar2, in->site.title);
    f[1].name = str_from_c("base_url");
    f[1].v = v_str(ar2, b.base);
    site_v = v_obj(ar2, f, 2);
  }
  Val *all_tags_v;
  {
    Val **items = NULL;
    if (ntags > 0) items = (Val **)ar_alloc(ar2, (size_t)ntags * sizeof(Val *));
    for (int k = 0; k < ntags; k++) {
      Field *f = (Field *)ar_alloc(ar2, 4 * sizeof(Field));
      f[0].name = str_from_c("name");
      f[0].v = v_str(ar2, tags[k]->name);
      f[1].name = str_from_c("slug");
      f[1].v = v_str(ar2, tags[k]->slug);
      f[2].name = str_from_c("url");
      f[2].v = v_str(ar2, tags[k]->url);
      f[3].name = str_from_c("count");
      f[3].v = v_num(ar2, tags[k]->nposts);
      items[k] = v_obj(ar2, f, 4);
    }
    all_tags_v = v_arr(ar2, items, ntags);
  }
  for (int k = 0; k < nposts; k++) {
    Field *cf = NULL;
    int ncf = 0;
    card_fields(&b, posts[k], &cf, &ncf);
    posts[k]->card = v_obj(ar2, cf, ncf);
  }
  Val *recent_v;
  {
    long long rc = in->site.recent;
    long long end;
    if (rc < 0) {
      end = (long long)nposts + rc;
      if (end < 0) end = 0;
    } else {
      end = rc < nposts ? rc : nposts;
    }
    Val **items = NULL;
    if (end > 0) items = (Val **)ar_alloc(ar2, (size_t)end * sizeof(Val *));
    for (long long k = 0; k < end; k++) items[k] = posts[k]->card;
    recent_v = v_arr(ar2, items, (int)end);
  }
  for (int i = 0; i < nkept; i++) {
    Source *s = kept[i];
    Val *pv = page_val(&b, s);
    Field vars[4];
    vars[0].name = str_from_c("site");
    vars[0].v = site_v;
    vars[1].name = str_from_c("tags");
    vars[1].v = all_tags_v;
    vars[2].name = str_from_c("recent");
    vars[2].v = recent_v;
    vars[3].name = str_from_c("page");
    vars[3].v = pv;
    Scope sc = {NULL, 0, {NULL, 0}, NULL, NULL, vars, 4};
    Str html;
    if (render_named(&b, s->layout, &sc, &html)) {
      Buf pb = {0};
      buf_addn(&pb, s->outdir.p, s->outdir.n);
      buf_adds(&pb, "/index.html");
      out_push(&b, buf_to_str(ar, &pb), html);
      buf_free(&pb);
    }
  }
  long long per = in->site.per_page;
  if (per < 1) per = 1;
  long long npages = (nposts + per - 1) / per;
  if (npages < 1) npages = 1;
  for (long long k = 1; k <= npages; k++) {
    long long a = (k - 1) * per, z = k * per;
    if (z > nposts) z = nposts;
    Val **items = NULL;
    if (z > a) items = (Val **)ar_alloc(ar2, (size_t)(z - a) * sizeof(Val *));
    for (long long t = a; t < z; t++) items[t - a] = posts[t]->card;
    Val *posts_v = v_arr(ar2, items, (int)(z - a));
    Field *pf = (Field *)ar_alloc(ar2, 5 * sizeof(Field));
    pf[0].name = str_from_c("number");
    pf[0].v = v_num(ar2, k);
    pf[1].name = str_from_c("total");
    pf[1].v = v_num(ar2, npages);
    pf[2].name = str_from_c("url");
    pf[2].v = v_str(ar2, index_url(ar2, b.base, k));
    pf[3].name = str_from_c("newer_url");
    pf[3].v = v_str(ar2, k == 1 ? str_from_c("") : index_url(ar2, b.base, k - 1));
    pf[4].name = str_from_c("older_url");
    pf[4].v = v_str(ar2, k == npages ? str_from_c("") : index_url(ar2, b.base, k + 1));
    Val *pager_v = v_obj(ar2, pf, 5);
    Field vars[5];
    vars[0].name = str_from_c("site");
    vars[0].v = site_v;
    vars[1].name = str_from_c("tags");
    vars[1].v = all_tags_v;
    vars[2].name = str_from_c("recent");
    vars[2].v = recent_v;
    vars[3].name = str_from_c("posts");
    vars[3].v = posts_v;
    vars[4].name = str_from_c("pager");
    vars[4].v = pager_v;
    Scope sc = {NULL, 0, {NULL, 0}, NULL, NULL, vars, 5};
    Str html;
    if (render_named(&b, str_from_c("index"), &sc, &html)) {
      Str path;
      if (k == 1) {
        path = str_from_c("index.html");
      } else {
        Buf pb = {0};
        char tmp[48];
        int m = snprintf(tmp, sizeof(tmp), "page/%lld/index.html", k);
        buf_addn(&pb, tmp, (size_t)m);
        path = buf_to_str(ar, &pb);
        buf_free(&pb);
      }
      out_push(&b, path, html);
    }
  }
  for (int k = 0; k < ntags; k++) {
    Val *tv = tag_val(&b, tags[k]);
    Field vars[4];
    vars[0].name = str_from_c("site");
    vars[0].v = site_v;
    vars[1].name = str_from_c("tags");
    vars[1].v = all_tags_v;
    vars[2].name = str_from_c("recent");
    vars[2].v = recent_v;
    vars[3].name = str_from_c("tag");
    vars[3].v = tv;
    Scope sc = {NULL, 0, {NULL, 0}, NULL, NULL, vars, 4};
    Str html;
    if (render_named(&b, str_from_c("tag"), &sc, &html)) {
      Buf pb = {0};
      buf_adds(&pb, "tags/");
      buf_addn(&pb, tags[k]->slug.p, tags[k]->slug.n);
      buf_adds(&pb, "/index.html");
      out_push(&b, buf_to_str(ar, &pb), html);
      buf_free(&pb);
    }
  }
  {
    Field vars[3];
    vars[0].name = str_from_c("site");
    vars[0].v = site_v;
    vars[1].name = str_from_c("tags");
    vars[1].v = all_tags_v;
    vars[2].name = str_from_c("recent");
    vars[2].v = recent_v;
    Scope sc = {NULL, 0, {NULL, 0}, NULL, NULL, vars, 3};
    Str html;
    if (render_named(&b, str_from_c("tags"), &sc, &html)) out_push(&b, str_from_c("tags/index.html"), html);
  }
  for (int i = 0; i < nstatics; i++) {
    FileEnt *f = statics_list[i];
    if (out_has(&b, f->path)) {
      Buf lb = {0};
      buf_adds(&lb, "shadowed ");
      buf_addn(&lb, f->path.p, f->path.n);
      buf_adds(&lb, ": a built page has its path");
      report_push(&b, buf_to_str(ar, &lb));
      buf_free(&lb);
    } else {
      out_push(&b, f->path, f->content);
    }
  }

  /* 9. The report. */
  {
    int total = b.nreport + mc->nbroken;
    Str *lines = NULL;
    if (total > 0) {
      lines = (Str *)ar_alloc(ar, (size_t)total * sizeof(Str));
      for (int i = 0; i < b.nreport; i++) lines[i] = b.report[i];
      for (int i = 0; i < mc->nbroken; i++) lines[b.nreport + i] = mc->broken[i];
      if (total > 1) {
        void *tmp = ar_alloc(ar, (size_t)total * sizeof(Str));
        msort_gen(lines, (size_t)total, sizeof(Str), tmp, cmp_str_el);
      }
    }
    int npages_built = 0;
    for (int i = 0; i < nkept; i++)
      if (!kept[i]->post) npages_built++;
    Buf rb = {0};
    for (int i = 0; i < total; i++) {
      buf_addn(&rb, lines[i].p, lines[i].n);
      buf_addc(&rb, '\n');
    }
    char tmp[128];
    int m = snprintf(tmp, sizeof(tmp), "built %d pages, %d posts, %d tags and %d static files\n", npages_built,
                     nposts, ntags, nstatics);
    buf_addn(&rb, tmp, (size_t)m);
    out_push(&b, str_from_c("build-report.txt"), buf_to_str(ar, &rb));
    buf_free(&rb);
  }

  /* Serialise the answer object, compact, like JSON.stringify. */
  buf_addc(out_json, '{');
  for (int i = 0; i < b.nout; i++) {
    if (i) buf_addc(out_json, ',');
    json_quote(out_json, b.out[i].path);
    buf_addc(out_json, ':');
    json_quote(out_json, b.out[i].text);
  }
  buf_addc(out_json, '}');

  smap_free(&b.tmap);
  smap_free(&b.by_src);
  smap_free(&b.statics);
  smap_free(&by_slug);
  smap_free(mc->broken_set);
  for (int i = 0; i < ntds; i++) smap_free(&tds[i].set);
}

/* ------------------------------------------------------------- N-API glue */

static napi_value throw_error(napi_env env, const char *msg) {
  napi_throw_error(env, NULL, msg);
  return NULL;
}

static int write_all(int fd, const void *p, size_t n) {
  const unsigned char *q = (const unsigned char *)p;
  while (n > 0) {
    ssize_t w = write(fd, q, n);
    if (w <= 0) return -1;
    q += (size_t)w;
    n -= (size_t)w;
  }
  return 0;
}

/* runOne(jsonText) -> the answer JSON string for one case's input. */
static napi_value js_run_one(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1] = {NULL};
  napi_status st = napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  if (st != napi_ok || argc < 1) return throw_error(env, "runOne needs one argument");
  size_t len = 0;
  st = napi_get_value_string_utf8(env, argv[0], NULL, 0, &len);
  if (st != napi_ok) return throw_error(env, "runOne needs a string argument");
  char *text = (char *)malloc(len + 1);
  if (!text) abort();
  napi_get_value_string_utf8(env, argv[0], text, len + 1, &len);

  Arena ar = {0};
  JP jp;
  memset(&jp, 0, sizeof(jp));
  jp.p = (const unsigned char *)text;
  jp.e = (const unsigned char *)text + len;
  jp.ar = &ar;
  CaseIn ci;
  jp_parse_input(&jp, &ci);
  napi_value result = NULL;
  if (!jp.err) {
    Buf out = {0};
    build_case(&ar, &ci, &out);
    napi_create_string_utf8(env, (const char *)out.b, out.len, &result);
    buf_free(&out);
  }
  buf_free(&jp.scratch);
  arena_free(&ar);
  free(text);
  if (jp.err) return throw_error(env, "invalid JSON input");
  return result;
}

/* runAll(casesText, outputPath) -> bytes written; writes answers.json. */
static napi_value js_run_all(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value argv[2] = {NULL, NULL};
  napi_status st = napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  if (st != napi_ok || argc < 2) return throw_error(env, "runAll needs two arguments");

  const unsigned char *text = NULL;
  size_t text_len = 0;
  char *text_copy = NULL;
  bool is_buffer = false;
  napi_is_buffer(env, argv[0], &is_buffer);
  if (is_buffer) {
    void *data = NULL;
    size_t blen = 0;
    napi_get_buffer_info(env, argv[0], &data, &blen);
    text = (const unsigned char *)data;
    text_len = blen;
  } else {
    size_t len = 0;
    st = napi_get_value_string_utf8(env, argv[0], NULL, 0, &len);
    if (st != napi_ok) return throw_error(env, "runAll needs a string or Buffer argument");
    text_copy = (char *)malloc(len + 1);
    if (!text_copy) abort();
    napi_get_value_string_utf8(env, argv[0], text_copy, len + 1, &len);
    text = (const unsigned char *)text_copy;
    text_len = len;
  }
  size_t plen = 0;
  st = napi_get_value_string_utf8(env, argv[1], NULL, 0, &plen);
  if (st != napi_ok) {
    free(text_copy);
    return throw_error(env, "runAll needs an output path string");
  }
  char *path = (char *)malloc(plen + 1);
  if (!path) abort();
  napi_get_value_string_utf8(env, argv[1], path, plen + 1, &plen);

  Arena ar = {0};
  JP jp;
  memset(&jp, 0, sizeof(jp));
  jp.p = text;
  jp.e = text + text_len;
  jp.ar = &ar;
  int ncases = 0;
  CaseIn *cases = jp_parse_cases(&jp, &ncases);
  napi_value result = NULL;
  int io_err = 0;
  size_t total = 0;
  if (!jp.err) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      io_err = 1;
    } else {
      if (write_all(fd, "[", 1) != 0) io_err = 1;
      total += 1;
      for (int i = 0; i < ncases && !io_err; i++) {
        Arena car = {0};
        Buf out = {0};
        build_case(&car, &cases[i], &out);
        if (i && write_all(fd, ",", 1) != 0) io_err = 1;
        else total += i ? 1 : 0;
        if (!io_err && write_all(fd, out.b, out.len) != 0) io_err = 1;
        else total += out.len;
        buf_free(&out);
        arena_free(&car);
      }
      if (!io_err && write_all(fd, "]", 1) != 0) io_err = 1;
      else total += 1;
      if (close(fd) != 0) io_err = 1;
      if (!io_err) napi_create_int64(env, (int64_t)total, &result);
    }
  }
  buf_free(&jp.scratch);
  arena_free(&ar);
  free(text_copy);
  free(path);
  if (jp.err) return throw_error(env, "invalid JSON input");
  if (io_err) return throw_error(env, "could not write the output file");
  return result;
}

static napi_value engine_init(napi_env env, napi_value exports) {
  napi_property_descriptor props[] = {
      {"runOne", NULL, js_run_one, NULL, NULL, NULL, napi_default, NULL},
      {"runAll", NULL, js_run_all, NULL, NULL, NULL, napi_default, NULL},
  };
  napi_status st = napi_define_properties(env, exports, 2, props);
  if (st != napi_ok) return NULL;
  return exports;
}

NAPI_MODULE(engine, engine_init)
