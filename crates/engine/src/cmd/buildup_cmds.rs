//! Build-up Layers: turns a flat pixel layer into the stack a painter would have built it from,
//! bottom to top: Foundation (big blurred masses of colour), Structure (the lightness of the
//! mid-scale forms), Color (the mid-scale colour), Texture (fine variation) and Detail (the
//! sharpest edges). The source layer is kept, hidden.
//!
//! The split is a frequency decomposition: with `b3 ≥ b2 ≥ b1` the image blurred by a large,
//! medium and small Gaussian, `Foundation = b3`, `mid = b2 - b3`, `fine = b1 - b2` and
//! `detail = src - b1`, so the bands sum back to the source. Each band is stored as a layer of
//! `0.5 + band / 2` in Linear Light (`base + 2 * layer - 1`), so the stack composites back to the
//! source and every layer can be faded or repainted on its own. Structure and Color are the
//! lightness and the colour parts of the mid band.

use photocraft_algo::matting::{box_mean, gaussian_blur};
use rayon::prelude::*;
use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::layer_place::{Place, insert_new};
use crate::{EngineError, Result, Session};

const CMD: &str = "layer.buildUpLayers";
/// Largest layer area (pixels) that is decomposed: the blurs hold about 50 bytes per pixel.
const MAX_PIXELS: usize = 40_000_000;

/// Names of the layers, bottom to top.
pub const LAYER_NAMES: [&str; 5] = ["Foundation", "Structure", "Color", "Texture", "Detail"];

fn check(doc: &Document, l: &Layer) -> std::result::Result<(), String> {
    if !matches!(l.content, LayerContent::Raster(_)) {
        return Err(format!("the layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()));
    }
    let locks = doc.effective_locks(l.id);
    if locks.all {
        return Err(format!("Could not complete your request because the layer \"{}\" is locked", l.name));
    }
    Ok(())
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    check(&d.doc, crate::active_layer_of(s)?)
}

/// Gaussian blur that costs the same at any radius: past a few pixels, three box passes (which
/// converge on a Gaussian) replace the kernel, whose cost grows with the radius.
fn blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 4.0 {
        return gaussian_blur(src, w, h, sigma);
    }
    // Three boxes of half-width r have variance 3 * ((2r + 1)^2 - 1) / 12.
    let r = (((4.0 * sigma * sigma + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
    let one = box_mean(src, w, h, r);
    let two = box_mean(&one, w, h, r);
    box_mean(&two, w, h, r)
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// A band value stored on a layer: `0.5` is "no change".
fn encode(d: f32) -> f32 {
    (0.5 + 0.5 * d).clamp(0.0, 1.0)
}

/// Splits straight RGBA pixels (`w × h`) into the five layers' pixels, bottom to top. `None` when
/// `cancelled` says so or the sizes disagree. Layers 2 to 5 are Linear Light offsets (see the
/// module docs); every layer keeps the source's alpha.
pub fn decompose(rgba: &[[f32; 4]], w: usize, h: usize, cancelled: &dyn Fn() -> bool) -> Option<[Vec<[f32; 4]>; 5]> {
    let n = w.checked_mul(h)?;
    if rgba.len() != n || n == 0 {
        return None;
    }
    // Large masses, medium forms, small forms (in pixels), always in decreasing order.
    let s3 = (w.max(h) as f32 * 0.02).max(2.0);
    let (s2, s1) = ((s3 / 4.0).max(1.0), (s3 / 16.0).max(0.5));
    let chans: Vec<Vec<f32>> = (0..3).map(|c| rgba.iter().map(|p| p.get(c).copied().unwrap_or(0.0)).collect()).collect();
    let mut blurs: Vec<[Vec<f32>; 3]> = Vec::with_capacity(3);
    for s in [s3, s2, s1] {
        if cancelled() {
            return None;
        }
        let three: Vec<Vec<f32>> = chans.par_iter().map(|ch| blur(ch, w, h, s)).collect();
        blurs.push(three.try_into().ok()?);
    }
    let (b3, b2, b1) = (blurs.first()?, blurs.get(1)?, blurs.get(2)?);
    let at = |v: &[Vec<f32>], c: usize, i: usize| v.get(c).and_then(|ch| ch.get(i)).copied().unwrap_or(0.0);
    let at3 = |b: &[Vec<f32>; 3], c: usize, i: usize| b.get(c).and_then(|ch| ch.get(i)).copied().unwrap_or(0.0);
    let mut out: [Vec<[f32; 4]>; 5] = std::array::from_fn(|_| Vec::with_capacity(n));
    for (i, src) in rgba.iter().enumerate() {
        let a = src.get(3).copied().unwrap_or(1.0);
        let mid: [f32; 3] = std::array::from_fn(|c| at3(b2, c, i) - at3(b3, c, i));
        let fine: [f32; 3] = std::array::from_fn(|c| at3(b1, c, i) - at3(b2, c, i));
        let det: [f32; 3] = std::array::from_fn(|c| at(&chans, c, i) - at3(b1, c, i));
        let lm = luma(mid);
        let layers = [
            [at3(b3, 0, i), at3(b3, 1, i), at3(b3, 2, i), a],
            [encode(lm), encode(lm), encode(lm), a],
            [encode(mid[0] - lm), encode(mid[1] - lm), encode(mid[2] - lm), a],
            [encode(fine[0]), encode(fine[1]), encode(fine[2]), a],
            [encode(det[0]), encode(det[1]), encode(det[2]), a],
        ];
        for (o, l) in out.iter_mut().zip(layers) {
            o.push(l);
        }
    }
    Some(out)
}

/// A layer of straight RGBA pixels over `area`, in `target`'s format.
fn layer_from(name: &str, rgba: &[[f32; 4]], area: Rect, target: PixelFormat) -> Result<Layer> {
    let mut work = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::F32, true));
    let flat: Vec<f32> = rgba.iter().flatten().copied().collect();
    work.write_region(area, &flat);
    let mut l = Layer::raster(name.to_string(), target);
    *crate::pixels_mut(&mut l)? = work.convert(target);
    Ok(l)
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    check(doc, layer).map_err(EngineError::Other)?;
    let area = layer.surface().map(|sf| sf.content_bounds().intersect(&doc.bounds())).unwrap_or(Rect::EMPTY);
    if area.is_empty() {
        return Err(EngineError::Other("the layer has no pixels to split".into()));
    }
    let (w, h) = (area.width() as usize, area.height() as usize);
    if w.checked_mul(h).is_none_or(|n| n > MAX_PIXELS) {
        return Err(EngineError::Other(format!("the layer is too large to split (more than {} megapixels)", MAX_PIXELS / 1_000_000)));
    }
    crate::jobs::edit_job(
        s,
        "Build-up Layers",
        move |doc, active, ctx| {
            let src = doc.layer(id).and_then(Layer::surface).ok_or(EngineError::NoLayer(id))?;
            let mut rgba = vec![[0.0f32; 4]; w * h];
            src.read_rgba_into(area, &mut rgba);
            ctx.progress(0.05, "Splitting into bands");
            let bands = decompose(&rgba, w, h, &|| ctx.cancelled()).ok_or(EngineError::Cancelled)?;
            let fmt = doc.pixel_format();
            let target = PixelFormat::new(fmt.mode, fmt.sample, true);
            let mut below = id;
            let mut ids = Vec::new();
            for (i, (name, px)) in LAYER_NAMES.iter().zip(bands.iter()).enumerate() {
                if ctx.cancelled() {
                    return Err(EngineError::Cancelled);
                }
                let mut l = layer_from(name, px, area, target)?;
                if i > 0 {
                    l.blend = BlendMode::LinearLight;
                }
                below = insert_new(doc, Some(below), l, Place::Above);
                ids.push(below.0);
            }
            doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.visible = false;
            *active = Some(below);
            Ok(ids)
        },
        move |ids| json!({ "layer": id.0, "layers": ids }),
    )
}

/// Build-up Layers command spec (Layer › New).
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Build-up Layers",
        menu: &["Layer", "New"],
        shortcut: None,
        params: r##"{"layer":id?} → {layer,layers:[id]} (Foundation, Structure, Color, Texture, Detail above the source, which is hidden; the stack composites back to the source; one history step)"##,
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_algo::segment::Rng;

    fn noisy(w: usize, h: usize) -> Vec<[f32; 4]> {
        let mut rng = Rng::new(7);
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let n = 0.04 * rng.normal();
                [
                    (0.5 + 0.4 * (x * 0.05).sin() + n).clamp(0.0, 1.0),
                    (0.4 + 0.3 * (y * 0.07).cos() + n).clamp(0.0, 1.0),
                    (0.6 + 0.2 * ((x + y) * 0.2).sin() + n).clamp(0.0, 1.0),
                    1.0,
                ]
            })
            .collect()
    }

    /// Stacks the bands the way the compositor does (Linear Light adds `2 * layer - 1`).
    fn stack(bands: &[Vec<[f32; 4]>; 5], i: usize) -> [f32; 3] {
        let mut c: [f32; 3] = std::array::from_fn(|k| bands[0][i][k]);
        for b in &bands[1..] {
            for k in 0..3 {
                c[k] = (c[k] + 2.0 * b[i][k] - 1.0).clamp(0.0, 1.0);
            }
        }
        c
    }

    #[test]
    fn the_bands_stack_back_to_the_source() {
        let (w, h) = (96, 72);
        let src = noisy(w, h);
        let bands = decompose(&src, w, h, &|| false).unwrap();
        let mut worst = 0.0f32;
        let mut sum = 0.0f32;
        for (i, p) in src.iter().enumerate() {
            let c = stack(&bands, i);
            for k in 0..3 {
                let e = (c[k] - p[k]).abs();
                worst = worst.max(e);
                sum += e;
            }
        }
        // Clipping between layers can cost a little at the extremes; on average it is exact.
        assert!(sum / (src.len() * 3) as f32 <= 0.002, "mean error {}", sum / (src.len() * 3) as f32);
        assert!(worst <= 0.1, "worst error {worst}");
    }

    #[test]
    fn each_band_holds_what_its_name_says() {
        let (w, h) = (96, 72);
        let src = noisy(w, h);
        let bands = decompose(&src, w, h, &|| false).unwrap();
        let spread = |b: &Vec<[f32; 4]>| {
            let m: f32 = b.iter().map(|p| p[0]).sum::<f32>() / b.len() as f32;
            (b.iter().map(|p| (p[0] - m).powi(2)).sum::<f32>() / b.len() as f32).sqrt()
        };
        // The foundation is smooth: far less variation between neighbours than the source.
        let jitter = |b: &Vec<[f32; 4]>| (0..b.len() - 1).filter(|i| i % w != w - 1).map(|i| (b[i][0] - b[i + 1][0]).abs()).sum::<f32>();
        assert!(jitter(&bands[0]) < jitter(&src) * 0.5);
        // Structure is gray (lightness only), Color has no lightness.
        assert!(bands[1].iter().all(|p| p[0] == p[1] && p[1] == p[2]));
        assert!(bands[2].iter().all(|p| (luma([p[0] - 0.5, p[1] - 0.5, p[2] - 0.5])).abs() < 1e-5));
        // Texture and Detail are centred on 50% gray.
        for b in [&bands[3], &bands[4]] {
            assert!(spread(b) > 0.0);
            let mean: f32 = b.iter().map(|p| p[0]).sum::<f32>() / b.len() as f32;
            assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
        }
    }

    #[test]
    fn decompose_rejects_bad_sizes_and_cancels() {
        let src = noisy(8, 8);
        assert!(decompose(&src, 8, 7, &|| false).is_none());
        assert!(decompose(&[], 0, 0, &|| false).is_none());
        assert!(decompose(&src, 8, 8, &|| true).is_none());
    }

    fn session(depth: u32) -> Session {
        let (w, h) = (96usize, 72usize);
        let src = noisy(w, h);
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::new();
            for p in &src {
                data.extend_from_slice(&p[..n.min(3)]);
                if n == 4 {
                    data.push(1.0);
                }
            }
            bg.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
            Ok(())
        })
        .unwrap();
        s
    }

    #[test]
    fn build_up_makes_five_layers_in_one_step_and_composites_back_at_every_depth() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let before = photocraft_compose::render(&s.active().unwrap().doc, Rect::new(0, 0, 96, 72));
            let (layers, steps) = (s.active().unwrap().doc.layers.len(), s.active().unwrap().history.past_len());
            assert!(s.is_enabled(CMD));
            let r = s.execute(CMD, json!({})).unwrap();
            assert_eq!(r["layers"].as_array().unwrap().len(), 5, "depth {depth}");
            let d = &s.active().unwrap().doc;
            assert_eq!(d.layers.len(), layers + 5, "depth {depth}");
            assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "depth {depth}: one history step");
            let names: Vec<&str> = d.layers.iter().skip(layers).map(|l| l.name.as_str()).collect();
            assert_eq!(names, LAYER_NAMES, "depth {depth}");
            assert!(d.layers.iter().skip(layers + 1).all(|l| l.blend == BlendMode::LinearLight));
            assert!(!d.layers[0].visible, "depth {depth}: the source is hidden");
            let after = photocraft_compose::render(d, Rect::new(0, 0, 96, 72));
            let err: f32 = before.px.iter().zip(&after.px).map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).abs()).sum::<f32>() / 3.0).sum::<f32>() / before.px.len() as f32;
            let tol = if depth == 8 { 0.02 } else { 0.01 };
            assert!(err <= tol, "depth {depth}: mean composite error {err}");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(s.active().unwrap().doc.layers.len(), layers, "depth {depth}");
            assert!(s.active().unwrap().doc.layers[0].visible, "depth {depth}: undo shows the source again");
        }
    }

    #[test]
    fn build_up_disabled_states_and_bad_params() {
        assert!(!Session::new().is_enabled(CMD));
        assert!(Session::new().execute(CMD, json!({})).is_err());

        let mut s = session(8);
        s.execute("type.create", json!({"text": "Hi", "size": 24, "x": 10, "y": 20})).unwrap();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());

        let mut s = session(8);
        let before = s.active().unwrap().revision;
        assert!(s.execute(CMD, json!({"layer": u64::MAX})).is_err());
        assert!(s.execute(CMD, json!({"layer": "x"})).is_err());
        assert_eq!(s.active().unwrap().revision, before);

        // An empty layer has nothing to split.
        let mut s = session(8);
        s.execute("layer.new.layer", json!({"name": "Empty"})).unwrap();
        let before = s.active().unwrap().doc.layers.len();
        assert!(s.execute(CMD, json!({})).is_err());
        assert_eq!(s.active().unwrap().doc.layers.len(), before);
    }
}
