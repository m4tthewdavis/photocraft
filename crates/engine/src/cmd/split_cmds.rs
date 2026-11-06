//! Split into Layers: peels the most prominent objects off a pixel layer onto layers of their
//! own and fills in what they leave behind, so a flat photo becomes a stack (objects over a
//! clean background) in one history step.
//!
//! This is the classical first stage: objects come from Select Subject (see `algo::segment`) and
//! the hole each one leaves is repaired by Content-Aware Fill, so the next round finds the next
//! most prominent object in the repaired layer. It finds no text and no effects; a learned
//! backend can replace the "find an object" step later without changing the command.

use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::{EngineError, Result, Session};

const CMD: &str = "layer.splitIntoLayers";
const DEFAULT_OBJECTS: usize = 3;
const MAX_OBJECTS: usize = 8;
/// Pixels the repaired area is grown by, so the fringe around a cut-out object is replaced too.
const FILL_GROW: u32 = 3;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

/// Splitting needs an unlocked pixel layer.
fn check(doc: &Document, l: &Layer) -> std::result::Result<(), String> {
    if !matches!(l.content, LayerContent::Raster(_)) {
        return Err(format!("the layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()));
    }
    let locks = doc.effective_locks(l.id);
    if locks.pixels || locks.all {
        return Err(format!("Could not complete your request because the layer \"{}\" is locked", l.name));
    }
    Ok(())
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    check(&d.doc, crate::active_layer_of(s)?)
}

/// `maxObjects`: how many objects to peel off at most (1..=8, default 3).
fn max_objects(p: &Value) -> Result<usize> {
    match p.get("maxObjects") {
        None | Some(Value::Null) => Ok(DEFAULT_OBJECTS),
        Some(v) => v
            .as_u64()
            .filter(|n| (1..=MAX_OBJECTS as u64).contains(n))
            .map(|n| n as usize)
            .ok_or_else(|| bad(format!("maxObjects must be a whole number from 1 to {MAX_OBJECTS}"))),
    }
}

fn set_active(s: &mut Session, id: LayerId) -> Result<()> {
    s.active_mut().ok_or(EngineError::NoDocument)?.active_layer = Some(id);
    Ok(())
}

fn active(s: &Session) -> Result<LayerId> {
    s.active().and_then(|d| d.active_layer).ok_or(EngineError::NoDocument)
}

/// One round: the most prominent object left on `src` goes onto a new layer above it and `src`
/// is repaired where it was. `None` when no object is found (nothing has changed then).
fn peel(s: &mut Session, src: LayerId, n: usize) -> Result<Option<LayerId>> {
    set_active(s, src)?;
    let found = s.execute("select.subject", json!({}))?;
    if found.get("changed").and_then(Value::as_bool) == Some(false) {
        return Ok(None);
    }
    let selection = s.active().ok_or(EngineError::NoDocument)?.doc.selection.clone();
    s.execute("layer.new.layerViaCopy", json!({}))?;
    let object = active(s)?;
    s.edit("Name object", |doc, _| {
        doc.layer_mut(object).ok_or(EngineError::NoLayer(object))?.name = format!("Object {n}");
        Ok(())
    })?;
    // The copy deselects; bring the selection back, grown, to repair the source under it.
    s.edit("Restore selection", |doc, _| {
        doc.selection = selection;
        Ok(())
    })?;
    s.execute("select.modify.expand", json!({ "radius": FILL_GROW }))?;
    set_active(s, src)?;
    s.execute("edit.contentAwareFill", json!({ "layer": src.0 }))?;
    s.execute("select.deselect", json!({}))?;
    Ok(Some(object))
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    check(doc, layer).map_err(EngineError::Other)?;
    let max = max_objects(p)?;
    crate::analysis_cmds::compound(s, "Split into Layers", |s| {
        let mut made = Vec::new();
        for n in 1..=max {
            match peel(s, id, n) {
                Ok(Some(object)) => made.push(object),
                Ok(None) => break,
                // A later round may fail to repair (nothing left to sample); keep what was found.
                Err(_) if !made.is_empty() => break,
                Err(e) => return Err(e),
            }
        }
        if made.is_empty() {
            return Err(EngineError::Other("no objects found to split off".into()));
        }
        set_active(s, id)?;
        Ok(json!({ "layer": id.0, "objects": made.iter().map(|l| l.0).collect::<Vec<_>>() }))
    })
}

/// Split into Layers command spec (Layer › New).
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Split into Layers",
        menu: &["Layer", "New"],
        shortcut: None,
        params: r##"{"layer":id?,"maxObjects":1..8=3} → {layer,objects:[id]} (the most prominent objects each onto a layer above, the hole in the source repaired; one history step)"##,
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_algo::segment::Rng;
    use photocraft_geom::Rect;
    use serde_json::json;

    /// A noisy blue background with one textured red disc (centre (100, 80), radius 45).
    fn disc_session(depth: u32) -> Session {
        let (w, h) = (200u32, 160u32);
        let mut rng = Rng::new(2);
        let nz: Vec<f32> = (0..(w * h) as usize).map(|_| rng.normal()).collect();
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let v = 0.05 * nz[(y * w + x) as usize];
                    let (dx, dy) = (x as f32 + 0.5 - 100.0, y as f32 + 0.5 - 80.0);
                    let c = if dx * dx + dy * dy <= 45.0 * 45.0 {
                        let t = ((x + y) as f32 * 0.35).sin() * 0.5 + 0.5;
                        [0.85 + 0.05 * t + v, 0.2 + 0.3 * t + v, 0.15 + v]
                    } else {
                        [0.25 + v, 0.45 + v, 0.7 + v]
                    };
                    data.extend(c.map(|c| c.clamp(0.0, 1.0)).iter().take(n.min(3)));
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
            Ok(())
        })
        .unwrap();
        s
    }

    fn count(s: &Session) -> usize {
        s.active().unwrap().doc.layers.len()
    }

    #[test]
    fn split_peels_the_subject_onto_its_own_layer_in_one_step_at_every_depth() {
        for depth in [8, 16, 32] {
            let mut s = disc_session(depth);
            assert!(s.is_enabled(CMD));
            let steps = s.active().unwrap().history.past_len();
            let r = s.execute(CMD, json!({"maxObjects": 1})).unwrap();
            assert_eq!(r["objects"].as_array().unwrap().len(), 1, "depth {depth}");
            assert_eq!(count(&s), 2, "depth {depth}: background plus one object");
            assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "depth {depth}: one history step");
            assert!(s.active().unwrap().doc.selection.is_none());
            let obj = r["objects"][0].as_u64().unwrap();
            let l = s.active().unwrap().doc.layers.iter().find(|l| l.id.0 == obj).unwrap();
            assert_eq!(l.name, "Object 1");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(count(&s), 1, "depth {depth}: undo removes the object layer");
            s.execute("edit.redo", json!({})).unwrap();
            assert_eq!(count(&s), 2, "depth {depth}: redo restores it");
        }
    }

    #[test]
    fn split_with_nothing_to_find_changes_nothing() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
        let (layers, steps) = (count(&s), s.active().unwrap().history.past_len());
        assert!(s.execute(CMD, json!({})).is_err());
        // A rolled-back compound restores the document (the revision counter still moves).
        assert_eq!(count(&s), layers);
        assert_eq!(s.active().unwrap().history.past_len(), steps);
        assert!(s.active().unwrap().doc.selection.is_none());
    }

    #[test]
    fn split_disabled_states_and_bad_params() {
        assert!(!Session::new().is_enabled(CMD));
        assert!(Session::new().execute(CMD, json!({})).is_err());

        let mut s = disc_session(8);
        s.execute("type.create", json!({"text": "Hi", "size": 24, "x": 10, "y": 20})).unwrap();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());

        let mut s = disc_session(8);
        s.execute("layer.setProps", json!({"locks": {"pixels": true}})).unwrap();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());

        let mut s = disc_session(8);
        let (before, layers) = (s.active().unwrap().revision, count(&s));
        for bad in [json!({"maxObjects": 0}), json!({"maxObjects": 99}), json!({"maxObjects": "many"}), json!({"maxObjects": -1}), json!({"layer": u64::MAX}), json!({"layer": "x"})] {
            assert!(s.execute(CMD, bad.clone()).is_err(), "{bad}");
        }
        assert_eq!(s.active().unwrap().revision, before);
        assert_eq!(count(&s), layers);
    }
}
