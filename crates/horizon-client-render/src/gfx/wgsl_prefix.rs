//! Copies of the renderer's own WGSL that run beside a shader pack's code in one module (the
//! sky and material stages of Horizon's packs, docs/shaders.md): every name such a copy
//! declares takes a prefix starting with `horizon_`, which the pack check refuses in a pack,
//! so no pack name can meet one of Horizon's.

/// terrain.wgsl, which the copies are taken from.
pub const TERRAIN: &str = include_str!("../terrain.wgsl");

/// The sprite sampling arms of terrain.wgsl's `sprite_grad` and `sprite_level`
/// and of the pack material's `horizon_block_sprite`, for `arrays` texture
/// arrays: `sprites` holds layers 0..SPRITE_SPLIT, `sprites_1` the next, and
/// so on. GLES binds at most 16 sampled textures to a stage, and the
/// terrain's group 0 spends one per array, so only the arrays that hold
/// layers are declared, sampled and bound: with one array the functions never
/// so much as mention `sprites_1`, which leaves the layout, the bind group
/// and the shader agreeing on the live count.
fn sprite_arms(call: &str, tail: &str, arrays: u32) -> String {
    let sample = |name: &str| {
        format!("        return {call}({name}, sprite_sampler, uv, l{tail});\n")
    };
    let array = |i: u32, out: &mut String| {
        if i == 0 {
            out.push_str("sprites");
        } else {
            out.push_str(&format!("sprites_{i}"));
        }
    };
    if arrays <= 1 {
        return format!("    return {call}(sprites, sprite_sampler, uv, l{tail});\n");
    }
    let mut out = String::from("    switch layer / SPRITE_SPLIT {\n");
    for i in 0..arrays - 1 {
        let mut name = String::new();
        array(i, &mut name);
        out += &format!("        case {i}u: {{\n{}        }}\n", sample(&name));
    }
    let mut last = String::new();
    array(arrays - 1, &mut last);
    out += &format!(
        "        default: {{\n{}        }}\n    }}\n",
        sample(&last)
    );
    out
}

/// The pack material's `horizon_block_sprite` switch (native_material's
/// entries), for `arrays` arrays: as `sprite_terrain`, only the arrays that
/// hold layers are sampled.
pub fn native_sprite_arms(arrays: u32) -> String {
    let sample = |name: &str| {
        format!("        return textureSampleLevel({name}, s, uv, l, lod);\n")
    };
    let array = |i: u32, out: &mut String| {
        if i == 0 {
            out.push_str("horizon_terrain_sprites");
        } else {
            out.push_str(&format!("horizon_terrain_sprites_{i}"));
        }
    };
    if arrays <= 1 {
        return format!("    return textureSampleLevel(horizon_terrain_sprites, s, uv, l, lod);\n");
    }
    let mut out = String::from("    switch idx / horizon_terrain_SPRITE_SPLIT {\n");
    for i in 0..arrays - 1 {
        let mut name = String::new();
        array(i, &mut name);
        out += &format!("        case {i}u: {{\n{}        }}\n", sample(&name));
    }
    let mut last = String::new();
    array(arrays - 1, &mut last);
    out += &format!(
        "        default: {{\n{}        }}\n    }}\n",
        sample(&last)
    );
    out
}

/// terrain.wgsl with its sprite sampling arms written for `arrays` arrays.
pub fn sprite_terrain(source: &str, arrays: u32) -> String {
    source
        .replace("HZ_SPRITE_GRAD_ARMS", &sprite_arms("textureSampleGrad", ", dx, dy", arrays))
        .replace("HZ_SPRITE_LEVEL_ARMS", &sprite_arms("textureSampleLevel", ", lod", arrays))
}

/// The text of `source` from the line starting with `start` up to (not including) the one
/// starting with `end`.
pub fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let from = source
        .find(&format!("\n{start}"))
        .expect("the source has the section")
        + 1;
    let to = from
        + source[from..]
            .find(&format!("\n{end}"))
            .expect("the source ends the section")
        + 1;
    &source[from..to]
}

/// The names `text` declares at its top level: functions, structs, constants, overrides and
/// module variables (bindings).
fn declared(text: &str) -> Vec<&str> {
    let mut names = Vec::new();
    for line in text.lines() {
        // Top level only: a function's own `var` and `let` stay as they are.
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        let mut line = line;
        while let Some(rest) = line.strip_prefix('@') {
            // An attribute: `@vertex`, `@group(0)`, `@binding(1)`.
            let end = rest.find(' ').unwrap_or(rest.len());
            line = rest[end..].trim_start();
        }
        let rest = ["fn ", "struct ", "const ", "override ", "var "]
            .iter()
            .find_map(|k| line.strip_prefix(k))
            .or_else(|| {
                // `var<uniform> name`, `var<storage, read> name`.
                let rest = line.strip_prefix("var<")?;
                Some(rest[rest.find('>')? + 1..].trim_start())
            });
        if let Some(rest) = rest {
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            names.push(&rest[..end]);
        }
    }
    names
}

/// `text` with every name it declares at its top level, and every use of one, prefixed by
/// `prefix`. Comments are rewritten too, which is harmless.
pub fn prefixed(text: &str, prefix: &str) -> String {
    let names = declared(text);
    let mut out = String::with_capacity(text.len() * 2);
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if names.contains(&word.as_str()) {
            out.push_str(prefix);
        }
        out.push_str(word);
        word.clear();
    };
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            flush(&mut word, &mut out);
            out.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_top_level_names_and_their_uses_only() {
        let text = "struct S { a: f32, }\n\
            @group(0) @binding(0) var<uniform> g: S;\n\
            @group(1) @binding(2) var<storage, read> list: array<u32>;\n\
            override N: u32 = 1u;\n\
            const K: f32 = 2.0;\n\
            @fragment fn f(x: f32) -> @location(0) vec4<f32> {\n    \
                var a = g.a * K;\n    \
                return vec4(a + f32(list[N]) + x);\n\
            }\n";
        let out = prefixed(text, "p_");
        assert_eq!(
            out,
            "struct p_S { a: f32, }\n\
            @group(0) @binding(0) var<uniform> p_g: p_S;\n\
            @group(1) @binding(2) var<storage, read> p_list: array<u32>;\n\
            override p_N: u32 = 1u;\n\
            const p_K: f32 = 2.0;\n\
            @fragment fn p_f(x: f32) -> @location(0) vec4<f32> {\n    \
                var a = p_g.a * p_K;\n    \
                return vec4(a + f32(p_list[p_N]) + x);\n\
            }\n"
        );
        // Every top-level declaration of terrain.wgsl is found.
        let names = declared(TERRAIN);
        for name in [
            "Globals",
            "globals",
            "sprites_7",
            "SPRITE_SPLIT",
            "block_slots",
            "fs_sky",
        ] {
            assert!(names.contains(&name), "{name}");
        }
        assert!(!names.contains(&"out"));
    }
}
