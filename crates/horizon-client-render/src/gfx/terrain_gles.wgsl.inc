// Legacy GLES sprite selection; included into terrain.wgsl, not a standalone
// shader. The arms name only the arrays that hold layers, however many there
// are (terrain_source's `gles_sprite_arms`), exactly as many as the pipeline
// layout and the bind group hold: a body that so much as mentions `sprites_1`
// makes every stage that reaches it require binding 5, and naga does not fold
// the runtime `SPRITE_ARRAYS == 1u` guard an earlier version of this file
// carried — a one-array world (any vanilla-sized one on a GLES phone) was
// refused for a binding its layout never had and crashed at startup.
fn sprite_grad(uv: vec2<f32>, layer: u32, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    let l = i32(layer % SPRITE_SPLIT);
    let array_index = layer / SPRITE_SPLIT;
HZ_GLES_SPRITE_GRAD_ARMS
}
fn sprite_level(uv: vec2<f32>, layer: u32, lod: f32) -> vec4<f32> {
    let l = i32(layer % SPRITE_SPLIT);
    let array_index = layer / SPRITE_SPLIT;
HZ_GLES_SPRITE_LEVEL_ARMS
}

// Adreno 304 V@269 crashes while linking a locally indexed RGSS offset array.
// The four original samples and their accumulation order are unchanged.
fn sample_rgss(uv: vec2<f32>, layer: u32, pixel: vec2<f32>, du: vec2<f32>, dv: vec2<f32>,
               texel_screen: vec2<f32>) -> vec4<f32> {
    let max_texel = max(texel_screen.x, texel_screen.y);
    let min_pixel = min(pixel.x, pixel.y);
    let blend = smoothstep(min_pixel, min_pixel * 2.0, max_texel);
    let du_len = length(du);
    let dv_len = length(dv);
    let effective = sqrt(min(du_len, dv_len) * max(du_len, dv_len));
    let exact = max(0.0, log2(effective / min_pixel));
    let low = floor(exact);
    let mip_blend = fract(exact);
    var sum_low = vec4(0.0);
    var sum_high = vec4(0.0);
    let p0 = uv + vec2<f32>(0.125, 0.375) * pixel;
    sum_low += sprite_level(p0, layer, low);
    sum_high += sprite_level(p0, layer, low + 1.0);
    let p1 = uv + vec2<f32>(-0.125, -0.375) * pixel;
    sum_low += sprite_level(p1, layer, low);
    sum_high += sprite_level(p1, layer, low + 1.0);
    let p2 = uv + vec2<f32>(0.375, -0.125) * pixel;
    sum_low += sprite_level(p2, layer, low);
    sum_high += sprite_level(p2, layer, low + 1.0);
    let p3 = uv + vec2<f32>(-0.375, 0.125) * pixel;
    sum_low += sprite_level(p3, layer, low);
    sum_high += sprite_level(p3, layer, low + 1.0);
    let rgss = mix(sum_low * 0.25, sum_high * 0.25, mip_blend);
    return mix(sample_nearest(uv, layer, pixel, du, dv, texel_screen), rgss, blend);
}
