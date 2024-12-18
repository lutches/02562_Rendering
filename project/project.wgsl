// ============================================================================
//                               Global Constants
// ============================================================================
const MAX_LEVEL = 20u;
const BSP_LEAF  = 3u;
const shader_matte    : u32 = 1;
const shader_reflect  : u32 = 2;
const shader_refract  : u32 = 3;
const shader_phong    : u32 = 4;
const shader_glossy   : u32 = 5;

const tex_scale = 0.02;
const d = 1.0f; // camera constant
const e = vec3f(277.0, 274.0, -545.0); // eye point
const p = vec3f(277.0, 274.0, 0.0);    // look at point
const u = vec3f(0.0, 1.0, 0.0);         // up vector
const dir_light_dir = vec3f(-0.3, -0.1, -0.8);

// ============================================================================
//                               Type Definitions
// ============================================================================
struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) coords: vec2f
};

struct UniformsF {
  aspect: f32,
  gamma: f32,
};

struct UniformsInt {
  canvas_width:  u32,
  canvas_height: u32,
  frame_num:     u32
};

struct AABB {
  min: vec3f,
  max: vec3f,
};

struct VertexAttrib {
  pos: vec4f,
  normal: vec4f,
};

struct Material {
  diffuse: vec4f,
  emission: vec4f,
};

struct Ray {
  origin: vec3f,
  direction: vec3f,
  tmin: f32,
  tmax: f32,
};

struct HitInfo {
  hit: bool,
  distance: f32,
  position: vec3f,
  normal: vec3f,
  color_amb: vec3f,
  color_diff: vec3f,
  color_specular: vec3f,
  shine: f32,
  refractive_ratio: f32,
  shader: u32,
  texcoords: vec2f,
};

struct Light {
  Li: vec3f,
  wi: vec3f,
  dist: f32,
};

struct FSOut {
  @location(0) frame: vec4f,
  @location(1) accum: vec4f,
};

// ============================================================================
//                               Bindings
// ============================================================================
@group(0) @binding(0) var<uniform> uniforms_f: UniformsF;
@group(0) @binding(1) var<uniform> uniforms_int: UniformsInt;
@group(0) @binding(3) var<uniform> aabb: AABB;
@group(0) @binding(4) var<storage> vert_attribs: array<VertexAttrib>;
@group(0) @binding(6) var<storage> materials: array<Material>;
@group(0) @binding(7) var<storage> vert_indices: array<vec4u>;
@group(0) @binding(8) var<storage> treeIds: array<u32>;
@group(0) @binding(9) var<storage> bspTree: array<vec4u>;
@group(0) @binding(10) var<storage> bspPlanes: array<f32>;
@group(0) @binding(11) var<storage> light_indices: array<u32>;
@group(0) @binding(12) var renderTexture: texture_2d<f32>;

// ============================================================================
//                               Utility Functions
// ============================================================================
fn tea(val0: u32, val1: u32) -> u32 {
  const N = 16u;
  var v0 = val1;
  var v1 = val0;
  var s0 = 0u;
  for(var n = 0u; n < N; n++){
    s0 += 0x9e3779b9u;
    v0 += ((v1<<4)+0xa341316cu)^(v1+s0)^((v1>>5)+0xc8013ea4u); 
    v1 += ((v0<<4)+0xad90777du)^(v0+s0)^((v0>>5)+0x7e95761eu); 
  }
  return v0;
}

fn mcg31(prev: ptr<function, u32>) -> u32 {
  const LCG_A = 1977654935u;
  *prev = (LCG_A * (*prev)) & 0x7FFFFFFFu;
  return *prev; 
}

// Generates a pseudo-random float in [0,1)
fn rnd(prev: ptr<function, u32>) -> f32 {
  return f32(mcg31(prev)) / f32(0x80000000u);
}

// Compute triangle area
fn triangle_area(a: vec3f, b: vec3f, c: vec3f) -> f32 {
  return length(cross(b - a, c - a)) * 0.5;
}

// Generate a camera ray given pixel coordinates
fn get_camera_ray(ipcoords: vec2f) -> Ray {
  let v = normalize(p - e);
  let b1 = normalize(cross(v, u));
  let b2 = cross(b1, v);

  let q = (v)*d + (b1 * ipcoords.x) + (b2 * ipcoords.y);
  return Ray(e, normalize(q), 1e-9, 1e9);
}

// ============================================================================
//                           Scene Intersection Functions
// ============================================================================
fn int_aabb(r: ptr<function, Ray>) -> bool {
  let p1 = (aabb.min - (*r).origin) / (*r).direction;
  let p2 = (aabb.max - (*r).origin) / (*r).direction;

  let pmin = min(p1, p2);
  let pmax = max(p1, p2);
  let tmin = max(pmin.x, max(pmin.y, pmin.z));
  let tmax = min(pmax.x, min(pmax.y, pmax.z));

  if (tmin > tmax || tmin > (*r).tmax || tmax < (*r).tmin) {
    return false;
  }

  (*r).tmin = max(tmin - 1e-3, (*r).tmin);
  (*r).tmax = min(tmax + 1e-3, (*r).tmax);
  return true;
}

var<private> branch_node: array<vec2u, MAX_LEVEL>; 
var<private> branch_ray:  array<vec2f, MAX_LEVEL>; 

fn int_triangle(ray: Ray, hit: ptr<function, HitInfo>, i: u32) -> bool {
  let verts = vert_indices[i].xyz;
  let v0 = vert_attribs[verts[0]].pos.xyz;
  let v1 = vert_attribs[verts[1]].pos.xyz;
  let v2 = vert_attribs[verts[2]].pos.xyz;

  let n0 = vert_attribs[verts[0]].normal.xyz;
  let n1 = vert_attribs[verts[1]].normal.xyz;
  let n2 = vert_attribs[verts[2]].normal.xyz;

  let e0 = v1 - v0;
  let e1 = v2 - v0;
  let n = cross(e0, e1);
  let denom = dot(ray.direction, n);

  if (abs(denom) < 1e-8) {
    return (*hit).hit; // no intersection if nearly parallel
  }

  let origin_to_v0 = v0 - ray.origin;
  let t = dot(origin_to_v0, n)/denom;
  if (t <= ray.tmax && t >= ray.tmin) {
    let partial = cross(origin_to_v0, ray.direction);
    let beta = dot(partial, e1) / denom;
    if (beta >= 0.0) {
      let gamma = -dot(partial, e0) / denom;
      if (gamma >= 0.0 && (beta + gamma) <= 1.0) {
        (*hit).hit = true;
        (*hit).distance = t;
        (*hit).position = ray.origin + t * ray.direction;

        let alpha = 1.0 - (beta + gamma);
        (*hit).normal = normalize(alpha*n0 + beta*n1 + gamma*n2);

        let matIndex = vert_indices[i].w;
        if (matIndex < arrayLength(&materials)) {
          let material = materials[matIndex];
          (*hit).color_diff = material.diffuse.rgb;
          (*hit).color_amb = material.emission.rgb;
        } else {
          (*hit).color_amb = vec3f(1.0); // fallback if invalid
          (*hit).color_diff = vec3f(1.0);
        }

        (*hit).shader = shader_matte;
      }
    }
  }

  return (*hit).hit;
}

fn int_trimesh(ray: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
  var branch_level = 0u;
  var node = 0u;

  for(var i = 0u; i <= MAX_LEVEL; i++) {
    let tree_node = bspTree[node];
    let node_axis_leaf = tree_node.x & 3u;

    // Leaf node
    if (node_axis_leaf == BSP_LEAF) {
      let node_count = tree_node.x >> 2u;
      let node_id = tree_node.y;
      var found = false;
      for (var j = 0u; j < node_count; j++) {
        let obj_idx = treeIds[node_id + j];
        if (int_triangle(*ray, hit, obj_idx)) {
          (*ray).tmax = (*hit).distance;
          found = true;
        }
      }
      if (found) { return true; }

      // Backtrack up the tree if no hit found in this leaf
      if (branch_level == 0u) {
        return false; // no more branches to check
      } else {
        branch_level -= 1u;
        i = branch_node[branch_level].x;
        node = branch_node[branch_level].y;
        (*ray).tmin = branch_ray[branch_level].x;
        (*ray).tmax = branch_ray[branch_level].y;
        continue;
      }
    }

    let axis = node_axis_leaf;
    let axis_direction = (*ray).direction[axis];
    let axis_origin    = (*ray).origin[axis];

    var near_node: u32;
    var far_node: u32;

    if (axis_direction >= 0.0) {
      near_node = tree_node.z;
      far_node  = tree_node.w;
    } else {
      near_node = tree_node.w;
      far_node  = tree_node.z;
    }

    let node_plane = bspPlanes[node];
    var denom = axis_direction;
    if (abs(denom) < 1e-8) {
      denom = select(1e-8, -1e-8, denom < 0.0);
    }

    let t = (node_plane - axis_origin) / denom;

    if (t >= (*ray).tmax) {
      node = near_node;
    } else if (t <= (*ray).tmin) {
      node = far_node;
    } else {
      branch_node[branch_level] = vec2u(i, far_node);
      branch_ray[branch_level] = vec2f(t, (*ray).tmax);
      branch_level += 1u;
      (*ray).tmax = t;
      node = near_node;
    }
  }
  return false;
}

fn int_aabb_scene(ray: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
  if (int_aabb(ray)) {
    return int_trimesh(ray, hit);
  }
  return false;
}

// Complete scene intersection
fn int_scene(ray: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
  return int_aabb_scene(ray, hit);
}

// ============================================================================
//                               Light Sampling
// ============================================================================

fn sample_light_triangle(index: u32, p: vec3f, seed: ptr<function, u32>) -> Light {
  let idx = light_indices[index];
  let vs = vert_indices[idx].xyz;
  let q0 = vert_attribs[vs[0]].pos.xyz;
  let q1 = vert_attribs[vs[1]].pos.xyz;
  let q2 = vert_attribs[vs[2]].pos.xyz;

  let n0 = vert_attribs[vs[0]].normal.xyz;
  let n1 = vert_attribs[vs[1]].normal.xyz;
  let n2 = vert_attribs[vs[2]].normal.xyz;

  // Barycentric sampling
  let r1 = rnd(seed);
  let r2 = rnd(seed);
  let alpha = 1.0 - sqrt(r1);
  let beta = (1.0 - r2)*sqrt(r1);
  let gamma = r2*sqrt(r1);

  let point = alpha*q0 + beta*q1 + gamma*q2;
  let norm = normalize(alpha*n0 + beta*n1 + gamma*n2);

  let dist = distance(p, point);
  let wi = normalize(point - p);

  let matIndex = vert_indices[idx].w;
  let Le = materials[matIndex].emission.rgb;

  // Area of the triangle
  let area = triangle_area(q0, q1, q2);

  // Li = Le * cos_theta * area / dist^2
  let Li = dot(-wi, norm) * Le * area * pow(1.0/dist, 2.0);

  return Light(Li, wi, dist);
}

fn sample_trimesh_light(p: vec3f, seed: ptr<function, u32>) -> Light {
  let numTriangles = arrayLength(&light_indices);
  var totalArea = 0.0;
  // Compute total area
  for (var i = 0u; i < numTriangles; i++) {
    let idx = light_indices[i];
    let vs = vert_indices[idx].xyz;
    let q0 = vert_attribs[vs[0]].pos.xyz;
    let q1 = vert_attribs[vs[1]].pos.xyz;
    let q2 = vert_attribs[vs[2]].pos.xyz;
    totalArea += length(cross(q1 - q0, q2 - q0));
  }

  let sampled_area = rnd(seed) * totalArea;
  var cumulArea = 0.0;
  var chosen = 0u;

  // Find which triangle to sample from
  for (var i = 0u; i < numTriangles; i++) {
    let idx = light_indices[i];
    let vs = vert_indices[idx].xyz;
    let q0 = vert_attribs[vs[0]].pos.xyz;
    let q1 = vert_attribs[vs[1]].pos.xyz;
    let q2 = vert_attribs[vs[2]].pos.xyz;
    let area = length(cross(q1 - q0, q2 - q0));
    cumulArea += area;
    if (cumulArea >= sampled_area) {
      chosen = i;
      break;
    }
  }

  return sample_light_triangle(chosen, p, seed);
}

fn check_shadow(pos: vec3f, lightdir: vec3f, lightdist: f32) -> bool {
  var lightray = Ray(pos, lightdir, 1e-3, lightdist-1e-3);
  var lighthit = HitInfo(false,0.0,vec3f(0),vec3f(0),vec3f(0),vec3f(0),vec3f(0),1.0,1.0,0,vec2f(0));
  return int_scene(&lightray, &lighthit);
}

// ============================================================================
//                               Shading Functions
// ============================================================================
fn lambert(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  var Lr = (*hit).color_amb / 3.14159;
  let light = sample_trimesh_light((*hit).position, seed);
  if (!check_shadow((*hit).position, light.wi, light.dist)) {
    Lr += ((*hit).color_diff / 3.14159) * light.Li * max(dot((*hit).normal, light.wi), 0.0);
  }
  return Lr;
}

fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  let wo = normalize((*r).origin - (*hit).position);
  var Lr = (*hit).color_amb / 3.14159;
  let light = sample_trimesh_light((*hit).position, seed);

  if (!check_shadow((*hit).position, light.wi, light.dist)) {
    let wr = reflect(light.wi, (*hit).normal);
    Lr += light.Li * dot(light.wi, (*hit).normal) *
          ( ((*hit).color_diff / 3.14159) +
            ((*hit).color_specular * ((*hit).shine + 2.0) * 0.15915494309 *
             pow(max(dot(wo, wr), 0.0), (*hit).shine)) );
  }
  return Lr;
}

fn shade_refract(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
  (*r).origin = (*hit).position;
  (*r).tmin = 1e-2;
  (*r).tmax = 1e6;
  (*hit).hit = false; 

  let n = (*hit).refractive_ratio;
  let cos_in = dot((*r).direction, (*hit).normal);
  let sin_sq_in = 1.0 - cos_in * cos_in;
  let sin_sq_out = (n * n) * sin_sq_in;
  let cos_sq_out = 1.0 - sin_sq_out;

  if(cos_sq_out < 0.0) {
    (*r).direction = reflect((*r).direction, (*hit).normal);
    return vec3f(1.0);
  }

  (*r).direction = (n * (*r).direction) - (n * cos_in + sqrt(cos_sq_out)) * (*hit).normal;
  return vec3f(0.0);
}

fn shade_reflect(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
  (*hit).hit = false;
  (*r).origin = (*hit).position;
  (*r).direction = reflect((*r).direction, (*hit).normal);
  (*r).tmin = 1e-2;
  (*r).tmax = 1e10;
  return vec3f(0.0);
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  switch (*hit).shader {
    case shader_matte   { return lambert(r, hit, seed); }
    case shader_reflect { return shade_reflect(r, hit); }
    case shader_refract { return shade_refract(r, hit); }
    case shader_phong   { return phong(r, hit, seed); }
    case shader_glossy  { return phong(r, hit, seed) + shade_refract(r, hit); }
    default             { return (*hit).color_diff + (*hit).color_amb; }
  }
}

// ============================================================================
//                               Entry Points
// ============================================================================

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut {
  const pos = array<vec2f, 4>(
    vec2f(-1, 1), vec2f(-1, -1),
    vec2f(1, 1),  vec2f(1, -1)
  );
  var vsOut: VSOut;
  vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
  vsOut.coords = pos[VertexIndex];
  return vsOut;
}

@fragment
fn main_fs(@builtin(position) fragcoord: vec4f, @location(0) coords: vec2f) -> FSOut {
  let launch_idx = u32(fragcoord.y)*uniforms_int.canvas_width + u32(fragcoord.x);
  var t = tea(launch_idx, uniforms_int.frame_num); 
  let jitter = vec2f(rnd(&t), rnd(&t)) / f32(uniforms_int.canvas_height);
  
  const bgcolor = vec4f(0.1, 0.3, 0.6, 0.9);
  const max_depth = 10;
  var result = vec3f(0.0);
  
  var hit = HitInfo(false,0.0,vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0),vec3f(0.0),1.0,1.0,0,vec2f(0.0));
  var ipcoords = vec2f((coords.x)*uniforms_f.aspect*0.5, (coords.y)*0.5);
  var r = get_camera_ray(ipcoords + jitter); 

  for(var i = 0; i < max_depth; i++) {
    if(int_scene(&r, &hit)) {
      result += shade(&r, &hit, &t);
      if (hit.hit || dot(result, result) >= 0.99) {
        break;
      }
    } else {
      result += bgcolor.rgb;
      break;
    }
  }

  let curr_sum = textureLoad(renderTexture, vec2u(fragcoord.xy), 0).rgb * f32(uniforms_int.frame_num);
  let accum_color = (result + curr_sum) / f32(uniforms_int.frame_num + 1u);

  var out: FSOut;
  out.frame = vec4f(pow(accum_color, vec3f(1.0 / uniforms_f.gamma)), 1.0);
  out.accum = vec4f(accum_color, 1.0);
  return out; 
}
