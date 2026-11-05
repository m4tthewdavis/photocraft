//! Shared helpers for the second filter batch: colour-model access, window
//! buffers, single-channel box/Gaussian blurs and jittered cell points.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::{from_rgba_into, to_rgba};

use crate::Ctx;
use crate::image::Image;
use crate::noise::hash01;

/// Max channels per pixel we keep on the stack (CMYK + alpha + spare).
pub(crate) const MAXC: usize = 8;

/// The context's model as an `f32` format (for colour conversions).
#[inline]
pub(crate) fn fmt(ctx: &Ctx) -> PixelFormat {
    PixelFormat::new(ctx.mode, SampleType::F32, ctx.alpha)
}

/// Colour channels (excluding alpha).
#[inline]
pub(crate) fn ncol(ctx: &Ctx, n: usize) -> usize {
    if ctx.alpha { n - 1 } else { n }
}

/// Whether channel values are ink amounts (1 = dark) rather than light.
#[inline]
pub(crate) fn subtractive(ctx: &Ctx) -> bool {
    ctx.mode == ColorMode::Cmyk
}

/// Whether per-channel "brightness" operations should run on an RGB
/// conversion instead of the native channels (Lab's a/b are not intensities).
#[inline]
pub(crate) fn via_rgb(ctx: &Ctx) -> bool {
    ctx.mode == ColorMode::Lab
}

/// Document coordinates of the `i`th pixel of `out`.
#[inline]
pub(crate) fn xy(out: Rect, i: usize) -> (i32, i32) {
    let w = out.width() as usize;
    (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32)
}

/// Straight sRGB RGBA of a native pixel.
#[inline]
pub(crate) fn rgba(ctx: &Ctx, px: &[f32]) -> [f32; 4] {
    to_rgba(&fmt(ctx), px)
}

/// Writes straight sRGB RGBA into a native pixel.
#[inline]
pub(crate) fn set_rgba(ctx: &Ctx, px: &mut [f32], c: [f32; 4]) {
    let mut tmp = [0.0f32; MAXC];
    let n = from_rgba_into(&fmt(ctx), c, &mut tmp);
    let m = n.min(px.len());
    px[..m].copy_from_slice(&tmp[..m]);
}

/// A straight sRGB RGBA colour in the native model.
pub(crate) fn native(ctx: &Ctx, c: [f32; 4]) -> [f32; MAXC] {
    let mut tmp = [0.0f32; MAXC];
    from_rgba_into(&fmt(ctx), c, &mut tmp);
    tmp
}

/// Rec. 601 luma of a native pixel.
#[inline]
pub(crate) fn luma(ctx: &Ctx, px: &[f32]) -> f32 {
    match ctx.mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => px[0],
        ColorMode::Lab => px[0],
        ColorMode::Rgb => 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2],
        _ => {
            let c = rgba(ctx, px);
            0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        }
    }
}

/// Interleaved premultiplied copy of `src` over `win` (zero outside).
pub(crate) fn premul_window(src: &Image, win: Rect, alpha: bool) -> Vec<f32> {
    let n = src.ch;
    let mut v = Vec::with_capacity(win.width() as usize * win.height() as usize * n);
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            let a = if alpha { src.get(x, y, n - 1) } else { 1.0 };
            for c in 0..n {
                let s = src.get(x, y, c);
                v.push(if alpha && c < n - 1 { s * a } else { s });
            }
        }
    }
    v
}

/// Unpremultiplies one interleaved pixel in place.
#[inline]
pub(crate) fn unpremul_px(px: &mut [f32], alpha: bool) {
    if !alpha {
        return;
    }
    let n = px.len();
    let a = px[n - 1];
    for v in px.iter_mut().take(n - 1) {
        *v = if a > 1e-7 { *v / a } else { 0.0 };
    }
}

/// The value an `f64` iterator `sum()` starts from. The running sums below fold their first
/// window by hand, starting here, so they keep the exact bits (signed zeros included) of the
/// `sum()` they replace.
#[inline]
fn f64_sum_start() -> f64 {
    std::iter::empty::<f64>().sum()
}

/// In-place running-sum box blur of every row of a `w × h × n` interleaved
/// buffer with radius `r` (edges clamped). `scratch` must hold `w × n`.
///
/// Per channel the arithmetic is the original's, in its order (first window summed left to right
/// in `f64`, then `acc += add − sub`), so results are bit-for-bit those of
/// [`reference::box_rows_n`]; only the clamped reads are confined to the row ends.
fn box_rows_n(buf: &mut [f32], w: usize, n: usize, r: usize, scratch: &mut Vec<f32>) {
    if r == 0 || w == 0 || n == 0 {
        return;
    }
    let norm = 1.0 / (2 * r + 1) as f64;
    for row in buf.chunks_exact_mut(w * n) {
        scratch.clear();
        scratch.extend_from_slice(row);
        // Fixed channel counts unroll the per-pixel channel loop.
        match n {
            1 => box_row::<1>(row, scratch, r, norm),
            2 => box_row::<2>(row, scratch, r, norm),
            3 => box_row::<3>(row, scratch, r, norm),
            4 => box_row::<4>(row, scratch, r, norm),
            _ => box_row_dyn(row, scratch, n, r, norm),
        }
    }
}

/// One row of [`box_rows_n`] for `N` channels: `src` is the row's original pixels.
fn box_row<const N: usize>(row: &mut [f32], src: &[f32], r: usize, norm: f64) {
    let px = src.as_chunks::<N>().0;
    let out = row.as_chunks_mut::<N>().0;
    let Some(last) = px.len().checked_sub(1) else { return };
    let mut acc = [f64_sum_start(); N];
    for x in -(r as isize)..=r as isize {
        let Some(p) = px.get(x.clamp(0, last as isize) as usize) else { return };
        for (a, &v) in acc.iter_mut().zip(p) {
            *a += v as f64;
        }
    }
    // Output x, then slide: add column x + r + 1, drop column x − r (both clamped).
    for (x, o) in out.iter_mut().enumerate() {
        let (Some(add), Some(sub)) = (px.get((x + r + 1).min(last)), px.get(x.saturating_sub(r))) else { return };
        for c in 0..N {
            o[c] = (acc[c] * norm) as f32;
            acc[c] += add[c] as f64 - sub[c] as f64;
        }
    }
}

/// [`box_row`] for any channel count.
fn box_row_dyn(row: &mut [f32], src: &[f32], n: usize, r: usize, norm: f64) {
    let w = src.len() / n;
    let Some(last) = w.checked_sub(1) else { return };
    let at = |x: usize, c: usize| src.get(x * n + c).copied().unwrap_or(0.0) as f64;
    let mut acc = vec![0.0f64; n];
    for (c, a) in acc.iter_mut().enumerate() {
        let mut sum = f64_sum_start();
        for x in -(r as isize)..=r as isize {
            sum += at(x.clamp(0, last as isize) as usize, c);
        }
        *a = sum;
    }
    for x in 0..w {
        let add = (x + r + 1).min(last);
        let sub = x.saturating_sub(r);
        let Some(out) = row.get_mut(x * n..(x + 1) * n) else { return };
        for (c, (o, a)) in out.iter_mut().zip(acc.iter_mut()).enumerate() {
            *o = (*a * norm) as f32;
            *a += at(add, c) - at(sub, c);
        }
    }
}

/// Same as [`box_rows_n`] over columns.
///
/// Walks the buffer a row at a time with one running sum per column and channel (sequential
/// memory access; the old per-column gather walked it with a stride of a row). The original rows
/// still to be subtracted are kept in a ring of `min(r + 1, h)` rows, so the extra memory is
/// bounded by the radius, not the image. Each column's arithmetic is unchanged, so results are
/// bit-for-bit those of [`reference::box_cols_n`].
fn box_cols_n(buf: &mut [f32], w: usize, h: usize, n: usize, r: usize, scratch: &mut Vec<f32>) {
    if r == 0 || h == 0 || w == 0 || n == 0 {
        return;
    }
    let stride = w * n;
    let Some(len) = stride.checked_mul(h) else { return };
    if buf.len() < len {
        return;
    }
    let norm = 1.0 / (2 * r + 1) as f64;
    let last = h - 1;
    // First window, summed top to bottom per column, as the original did.
    let mut acc = vec![f64_sum_start(); stride];
    for y in -(r as isize)..=r as isize {
        let yy = y.clamp(0, last as isize) as usize;
        let Some(src) = buf.get(yy * stride..(yy + 1) * stride) else { return };
        for (a, &v) in acc.iter_mut().zip(src) {
            *a += v as f64;
        }
    }
    // Ring of original rows: row k lives in slot k % rows while it can still be subtracted.
    let rows = (r + 1).min(h);
    scratch.clear();
    scratch.resize(rows * stride, 0.0);
    for y in 0..h {
        let slot = (y % rows) * stride;
        let (Some(keep), Some(row)) = (scratch.get_mut(slot..slot + stride), buf.get(y * stride..(y + 1) * stride)) else { return };
        keep.copy_from_slice(row);
        let add = (y + r + 1).min(last);
        let sub = y.saturating_sub(r);
        // `sub` ≤ y is always in the ring; `add` ≥ y is untouched in `buf` unless it is row y
        // itself (the last rows, clamped), which was just saved.
        let s0 = (sub % rows) * stride;
        let Some(sub_row) = scratch.get(s0..s0 + stride) else { return };
        let (head, tail) = buf.split_at_mut((y + 1) * stride);
        let Some(out) = head.get_mut(y * stride..) else { return };
        if add > y {
            let off = (add - y - 1) * stride;
            let Some(add_row) = tail.get(off..off + stride) else { return };
            for (((o, a), &av), &sv) in out.iter_mut().zip(acc.iter_mut()).zip(add_row).zip(sub_row) {
                *o = (*a * norm) as f32;
                *a += av as f64 - sv as f64;
            }
        } else {
            // add == y: its original is in the ring (the slot just written).
            let Some(add_row) = scratch.get(slot..slot + stride) else { return };
            for (((o, a), &av), &sv) in out.iter_mut().zip(acc.iter_mut()).zip(add_row).zip(sub_row) {
                *o = (*a * norm) as f32;
                *a += av as f64 - sv as f64;
            }
        }
    }
}

/// The straightforward running-sum passes [`box_rows_n`] and [`box_cols_n`] replaced, kept as
/// the oracle their outputs must match bit for bit.
#[cfg(test)]
mod reference {
    pub(super) fn box_rows_n(buf: &mut [f32], w: usize, n: usize, r: usize, scratch: &mut Vec<f32>) {
        if r == 0 || w == 0 {
            return;
        }
        let norm = 1.0 / (2 * r + 1) as f64;
        for row in buf.chunks_exact_mut(w * n) {
            scratch.clear();
            scratch.extend_from_slice(row);
            let at = |x: isize, c: usize| scratch[(x.clamp(0, w as isize - 1) as usize) * n + c] as f64;
            for c in 0..n {
                let mut acc: f64 = (-(r as isize)..=r as isize).map(|x| at(x, c)).sum();
                for x in 0..w {
                    row[x * n + c] = (acc * norm) as f32;
                    acc += at(x as isize + r as isize + 1, c) - at(x as isize - r as isize, c);
                }
            }
        }
    }

    pub(super) fn box_cols_n(buf: &mut [f32], w: usize, h: usize, n: usize, r: usize, scratch: &mut Vec<f32>) {
        if r == 0 || h == 0 {
            return;
        }
        let norm = 1.0 / (2 * r + 1) as f64;
        for x in 0..w {
            scratch.clear();
            for y in 0..h {
                scratch.extend_from_slice(&buf[(y * w + x) * n..(y * w + x + 1) * n]);
            }
            let at = |y: isize, c: usize| scratch[(y.clamp(0, h as isize - 1) as usize) * n + c] as f64;
            for c in 0..n {
                let mut acc: f64 = (-(r as isize)..=r as isize).map(|y| at(y, c)).sum();
                for y in 0..h {
                    buf[(y * w + x) * n + c] = (acc * norm) as f32;
                    acc += at(y as isize + r as isize + 1, c) - at(y as isize - r as isize, c);
                }
            }
        }
    }
}

#[cfg(test)]
mod box_tests {
    use super::{box_cols_n, box_rows_n, reference};

    fn data(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9) | 1;
        (0..len)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                match i % 11 {
                    // Signed zeros, a large value and exact ties among the noise.
                    0 => -0.0,
                    1 => 0.0,
                    2 => 1.0e6,
                    _ => (x >> 8) as f32 / 16_777_216.0 * 2.0 - 0.5,
                }
            })
            .collect()
    }

    fn same_bits(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
        }
    }

    #[test]
    fn passes_match_the_reference_bit_for_bit() {
        let mut seed = 1;
        for (w, h) in [(1, 1), (1, 9), (9, 1), (2, 3), (7, 5), (33, 17), (64, 40)] {
            for n in [1, 2, 3, 4, 5] {
                for r in [1, 2, 3, 6, 16, 40, 100] {
                    seed += 1;
                    let src = data(w * h * n, seed);
                    let (mut a, mut b) = (src.clone(), src.clone());
                    let (mut sa, mut sb) = (Vec::new(), Vec::new());
                    box_rows_n(&mut a, w, n, r, &mut sa);
                    reference::box_rows_n(&mut b, w, n, r, &mut sb);
                    same_bits(&a, &b, &format!("rows {w}x{h} n{n} r{r}"));
                    let (mut a, mut b) = (src.clone(), src);
                    box_cols_n(&mut a, w, h, n, r, &mut sa);
                    reference::box_cols_n(&mut b, w, h, n, r, &mut sb);
                    same_bits(&a, &b, &format!("cols {w}x{h} n{n} r{r}"));
                }
            }
        }
    }

    #[test]
    fn a_short_buffer_is_left_alone() {
        let mut buf = vec![0.25f32; 10];
        let before = buf.clone();
        box_cols_n(&mut buf, 4, 4, 1, 1, &mut Vec::new());
        assert_eq!(buf, before);
    }
}

/// Box blur of an interleaved buffer (radius `r` in both directions).
pub(crate) fn box_blur_n(buf: &mut [f32], w: usize, h: usize, n: usize, r: usize) {
    let mut scratch = Vec::with_capacity(w.max(h) * n);
    box_rows_n(buf, w, n, r, &mut scratch);
    box_cols_n(buf, w, h, n, r, &mut scratch);
}

/// Odd box widths whose 3-fold convolution approximates a Gaussian of `sigma`
/// (Kovesi 2010), returned as radii.
pub(crate) fn gauss_box_radii(sigma: f32) -> [usize; 3] {
    let n = 3.0f32;
    let w_ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i32;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let m = ((12.0 * sigma * sigma - n * (wl * wl) as f32 - 4.0 * n * wl as f32 - 3.0 * n) / (-4.0 * wl as f32 - 4.0)).round().clamp(0.0, n) as usize;
    let mut radii = std::array::from_fn(|i| if i < m { (wl as usize - 1) / 2 } else { (wu as usize - 1) / 2 });
    // Kovesi's rounded triple degenerates below sigma ~ 0.577: `wl` floors to 1 and every
    // radius becomes 0, so the passes would be no-ops and callers with documented small
    // sigmas (Camera Raw `sharpenRadius` 0.5, colour noise, HDR Toning `radius` 1) would
    // silently change nothing (issue #706). One radius-1 pass is the smallest real blur;
    // it only engages where the ideal widths are already at their floor, leaving larger
    // sigmas exactly as the paper prescribes.
    if radii.iter().all(|&r| r == 0) {
        radii[0] = 1;
    }
    radii
}

/// Approximate Gaussian blur (three box passes) of an interleaved buffer.
/// Exact small kernels are not needed here: callers use it for smoothing
/// fields and blur levels, where cost independent of `sigma` matters more.
pub(crate) fn gauss_blur_n(buf: &mut [f32], w: usize, h: usize, n: usize, sigma: f32) {
    if sigma < 0.3 {
        return;
    }
    let mut scratch = Vec::with_capacity(w.max(h) * n);
    for r in gauss_box_radii(sigma) {
        box_rows_n(buf, w, n, r, &mut scratch);
    }
    for r in gauss_box_radii(sigma) {
        box_cols_n(buf, w, h, n, r, &mut scratch);
    }
}

/// Jittered feature point of grid cell `(cx, cy)` (cells of `cell` px anchored at `origin`).
#[inline]
pub(crate) fn cell_point(cx: i32, cy: i32, cell: f32, seed: u32, origin: (f32, f32)) -> (f32, f32) {
    (origin.0 + (cx as f32 + 0.1 + 0.8 * hash01(cx, cy, 11, seed)) * cell, origin.1 + (cy as f32 + 0.1 + 0.8 * hash01(cx, cy, 12, seed)) * cell)
}

#[inline]
pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Adds the premultiplied bilinear sample of `src` at `(x, y)` (pixel centres at .5, edges
/// clamped to `clip`, which must lie inside `src.rect`) into `acc`. A fast path for blurs that
/// average many samples; unpremultiply the sum once at the end.
#[inline]
pub(crate) fn add_sample_premul(src: &Image, x: f32, y: f32, clip: Rect, alpha: bool, acc: &mut [f32]) {
    if !x.is_finite() || !y.is_finite() || clip.is_empty() {
        return;
    }
    let n = src.ch;
    let fx = (x - 0.5).clamp(clip.x0 as f32, (clip.x1 - 1) as f32);
    let fy = (y - 0.5).clamp(clip.y0 as f32, (clip.y1 - 1) as f32);
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (x1, y1) = ((x0 + 1).min(clip.x1 - 1), (y0 + 1).min(clip.y1 - 1));
    let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
    let w = src.rect.width() as usize;
    let idx = |xx: i32, yy: i32| ((yy - src.rect.y0) as usize * w + (xx - src.rect.x0) as usize) * n;
    for (i, k) in [(idx(x0, y0), (1.0 - ax) * (1.0 - ay)), (idx(x1, y0), ax * (1.0 - ay)), (idx(x0, y1), (1.0 - ax) * ay), (idx(x1, y1), ax * ay)] {
        if k <= 0.0 {
            continue;
        }
        let px = &src.data[i..i + n];
        let a = if alpha { px[n - 1] * k } else { k };
        if alpha {
            for c in 0..n - 1 {
                acc[c] += px[c] * a;
            }
            acc[n - 1] += a;
        } else {
            for c in 0..n {
                acc[c] += px[c] * a;
            }
        }
    }
}
