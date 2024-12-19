struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) coords: vec2f,
};

struct UniformsF { // float values
  aspect: f32,
  gamma: f32,
}

struct UniformsInt { // unsigned int values
  canvas_width: u32,
  canvas_height: u32,
  frame_num: u32,
  shader: u32,
}

// Axis-Aligned Bounding Box (box defined by min and max points)
struct AABB {
  min: vec3f,
  max: vec3f,
};

struct VertexAttrib {
  pos: vec4f,
  normal: vec4f,
}

struct Material {
  diffuse: vec4f,
  emission: vec4f,
}

struct Ray {
  origin: vec3f,
  direction: vec3f,
  tmin: f32,
  tmax: f32,
}

struct HitInfo {
  hit: bool,
  distance: f32,
  position: vec3f,
  normal: vec3f,
  color_amb: vec3f,      // ambient component of color
  color_diff: vec3f,     // diffuse (Lambertian) component
  color_specular: vec3f, // specular reflectance
  shine: f32,            // Phong exponent
  refractive_ratio: f32, // ratio of refractive indices
  shader: u32,
  texcoords: vec2f,      // texture coordinates at hit
}

struct Onb {
  tangent: vec3f,
  binormal: vec3f,
  normal: vec3f,
};

struct Light {
  Li: vec3f,
  wi: vec3f,
  dist: f32,
}
struct FSOut {
  @location(0) frame: vec4f,
  @location(1) accum: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms_f: UniformsF;
@group(0) @binding(1) var<uniform> uniforms_int: UniformsInt;
@group(0) @binding(3) var<uniform> aabb: AABB;
@group(0) @binding(4) var<storage> vert_attribs: array<VertexAttrib>;
@group(0) @binding(6) var<storage> materials: array<Material>; // colors
@group(0) @binding(7) var<storage> vert_indices: array<vec4u>;
@group(0) @binding(8) var<storage> treeIds: array<u32>;
@group(0) @binding(9) var<storage> bspTree: array<vec4u>;
@group(0) @binding(10) var<storage> bspPlanes: array<f32>;
//@group(0) @binding(11) var<storage> light_indices: array<u32>; // not used with this object
@group(0) @binding(12) var renderTexture: texture_2d<f32>;
@group(0) @binding(13) var bg_texture: texture_2d<f32>;

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut {
  const pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(1, -1));
  var vsOut: VSOut;
  vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
  vsOut.coords = pos[VertexIndex];
  return vsOut;
}



// shader type names:
const shader_reflect: u32 = 2;
const shader_refract: u32 = 3;
const shader_phong: u32 = 4;
const shader_glossy: u32 = 5;
const shader_matte: u32 = 1;
const shader_transparent: u32 = 6;

fn tea(val0: u32, val1: u32) -> u32 {
  const N = 16u;
  var v0 = val1;
  var v1 = val0;
  var s0 = 0u;
  for(var n = 0u; n < N; n ++){
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4); 
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e); 
  }
  return v0;
}

fn mcg31(prev: ptr<function, u32>) -> u32 {
  const LCG_A = 1977654935u; //  from Hui-Ching Tang [EJOR 2007
  *prev = (LCG_A * (*prev)) & 0x7FFFFFFF;
  return *prev; 
}

// generates pseudo random f32 in range [0, 1)
fn rnd(prev: ptr<function, u32>) -> f32 {
  return f32(mcg31(prev)) / f32(0x80000000);
}

const e = vec3f(-1, 10, 23);
const d = 1f;
const u = vec3f(0, 1, 0);
const p = vec3f(-1, 5, 0.0);
const dir_light_dir = vec3f(20, -8, 0.5);
const pi = 3.14159265359;



fn get_camera_ray(ipcoords: vec2f) -> Ray {
  let v = normalize(p - e);
  let b1 = normalize(cross(v, u));
  let b2 = cross(b1, v);
  
  let x = ipcoords[0];
  let y = ipcoords[1];
  let q = (v)*d + (b1 * x) + (b2 * y);
  var dir = normalize(q);
  return Ray(e, dir, 1e-9, 1e9);
}

fn int_scene(ray: ptr<function, Ray>,  hit: ptr<function, HitInfo>) -> bool {

  if(int_aabb(ray)){
    if(int_trimesh(ray, hit)){
      (*ray).tmax = (*hit).distance;
    }
  }

  return (*hit).hit;
}

fn int_skybox(ray: Ray) -> vec2f {
  var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 1.0, 1.0, 0, vec2f(0.0));
  const center = vec3f(0.0);
  const radius = 1e2;
  const shade_mode = shader_matte;

  int_sphere(ray, &hit, center, radius, shade_mode);

  let D = -hit.normal;

  let u = 0.5 + (atan2(D.x, -D.z) / (2*pi));
  let v = (acos(-D.y) / (pi));

  return vec2f(u, v);
}

fn sample_skybox(r: Ray) -> vec3f {
  return sample_skybox_light(r);
}

fn sample_skybox_light(r: Ray) -> vec3f {
  let uv = int_skybox(r);
  let s = texture_linear(bg_texture, uv);
  return s.rgb;
}

fn int_plane(ray: Ray, hit: ptr<function, HitInfo>, plane_point: vec3f, plane: Onb) -> bool {
  let omega_dot_n = dot(ray.direction, plane.normal);
  if (abs(omega_dot_n) > 1e-4) {
    let t = dot((plane_point - ray.origin), plane.normal) / omega_dot_n;
    if (ray.tmax >= t && ray.tmin <= t) {
      (*hit).hit = true;
      (*hit).distance = t;
      (*hit).position = ray.origin + t * ray.direction;
      (*hit).normal = plane.normal;
    }
  }
  return (*hit).hit;
}

fn holdout_occlusion(ray: Ray, seed: ptr<function, u32>) -> f32 {
  // If ray hits holdout plane, calculate ambient occlusion
  var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0),
                    vec3f(0.0), vec3f(0.0), 1.0, 1.0, 0, vec2f(0.0));
  let p = aabb.min;
  let onb = Onb(vec3f(0, 1, 0), vec3f(0, 0, 1), vec3f(0, 1, 0));
  
  if (!int_plane(ray, &hit, p, onb)) {
    return 1f;
  }

  if (hit.distance > 85) {
    return 1f;
  }

  let sample_cos_theta = sqrt(1f - rnd(seed));
  let sample_phi = 2 * pi * rnd(seed);
  let sample_sin_theta = sqrt(1f - sample_cos_theta * sample_cos_theta);
  let dir = rotate_to_normal(
    vec3f(0, 1, 0),
    spherical_direction(sample_sin_theta, sample_cos_theta, sample_phi)
  );

  var result = 0f;

  if (!check_shadow(hit.position, dir, 1e1)) {
    let s = sample_skybox(Ray(hit.position, dir, 1e-1, 1e9));
    result += length(s);
  }

  let dirLight = sample_directional_light(hit.position);
  if (!check_shadow(hit.position, dirLight.wi, dirLight.dist)) {
    result += length(dirLight.Li)/3;
  }

  return result / 2.5;
}

fn int_sphere(ray: Ray, hit: ptr<function, HitInfo>, center: vec3f, radius: f32, shade_mode: u32) -> bool {
  // Sphere intersection
  let dist_from_origin = ray.origin - center;
  let half_b = dot(dist_from_origin, ray.direction);
  let c = dot(dist_from_origin, dist_from_origin) - (radius * radius);
  let desc = (half_b * half_b) - c;

  if (desc < 0) {
    return (*hit).hit;
  }

  let t = -half_b;
  if (desc <= 1e-4) {
    // Tangent hit
    if (ray.tmax >= t && ray.tmin <= t) {
      (*hit).color_diff = vec3f(0) * 0.9;
      (*hit).color_amb = vec3f(0) * 0.1;
      (*hit).color_specular = vec3f(0.1);
      (*hit).shine = 42;
      (*hit).hit = true;
      (*hit).distance = t;
      let pos = ray.origin + t * ray.direction;
      (*hit).position = pos;
      (*hit).normal = normalize(pos - center);
      (*hit).shader = uniforms_int.shader;
      (*hit).refractive_ratio = 1.0;
    }
  } else {
    let sqrt_desc = sqrt(desc);
    let t1 = t - sqrt_desc;
    let t2 = t + sqrt_desc;

    if (ray.tmax >= t1 && ray.tmin <= t1) {
      (*hit).distance = t1;
      (*hit).hit = true;
    } else if (ray.tmax >= t2 && ray.tmin <= t2) {
      (*hit).distance = t2;
      (*hit).hit = true;
    }

    if ((*hit).hit) {
      let pos = ray.origin + ((*hit).distance * ray.direction);
      (*hit).position = pos;
      (*hit).color_amb = vec3f(0) * 0.1;
      (*hit).color_diff = vec3f(0) * 0.9;
      (*hit).color_specular = vec3f(0.1);
      (*hit).shine = 42;
      (*hit).normal = normalize((*hit).position - center);
      (*hit).shader = shade_mode;

      let refr_enter = 0.667;
      let refr_exit = 1.5;
      (*hit).refractive_ratio = refr_enter;
      if (dot(ray.direction, (*hit).normal) > 0) {
        (*hit).normal = -(*hit).normal;
        (*hit).refractive_ratio = refr_exit;
      }
    }
  }
  return (*hit).hit;
}

fn int_aabb(r: ptr<function, Ray>) -> bool {
  let p1 = (aabb.min - r.origin) / r.direction;
  let p2 = (aabb.max - r.origin) / r.direction;

  let pmin = min(p1, p2);
  let pmax = max(p1, p2);
  let tmin = max(pmin.x, max(pmin.y, pmin.z));
  let tmax = min(pmax.x, min(pmax.y, pmax.z));
  
  if (tmin > tmax || tmin > r.tmax || tmax < r.tmin) {
    return false;
  }

  r.tmin = max(tmin - 1e-3f, r.tmin);
  r.tmax = min(tmax + 1e-3f, r.tmax);
  return true;
}

const MAX_LEVEL = 20u;
const BSP_LEAF = 3u;
var<private> branch_node: array<vec2u, MAX_LEVEL>;
var<private> branch_ray: array<vec2f, MAX_LEVEL>;

fn int_trimesh(ray: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
  var branch_level = 0u;
  var near_node = 0u;
  var far_node = 0u;
  var t = 0.0f;
  var node = 0u;

  for (var i = 0u; i <= MAX_LEVEL; i++) {
    let tree_node = bspTree[node];
    let node_axis_leaf = tree_node.x & 3u;

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
      if (found) { 
        return true;
      } else if (branch_level == 0u) {
        return false;
      } else {
        branch_level--;
        i = branch_node[branch_level].x;
        node = branch_node[branch_level].y;
        (*ray).tmin = branch_ray[branch_level].x;
        (*ray).tmax = branch_ray[branch_level].y;
        continue;
      }
    }

    let axis_direction = (*ray).direction[node_axis_leaf];
    let axis_origin = (*ray).origin[node_axis_leaf];

    if (axis_direction >= 0.0) {
      near_node = tree_node.z; 
      far_node = tree_node.w;  
    } else {
      near_node = tree_node.w; 
      far_node = tree_node.z;  
    }

    let node_plane = bspPlanes[node];
    var denom = axis_direction;
    if (abs(denom) < 1e-8) {
      if(denom < 0) {denom = -1e-8;} else {denom = 1e-8;};
    }

    t = (node_plane - axis_origin) / denom;
    if (t >= (*ray).tmax) {
      node = near_node;
    } else if (t <= (*ray).tmin) {
      node = far_node;
    } else {
      branch_node[branch_level].x = i;
      branch_node[branch_level].y = far_node;
      branch_ray[branch_level].x = t;
      branch_ray[branch_level].y = (*ray).tmax;
      branch_level++;
      (*ray).tmax = t;
      node = near_node;
    }
  }

  return false;
}

fn int_triangle(ray: Ray, hit: ptr<function, HitInfo>, i: u32) -> bool {
  let verts = vert_indices[i].xyz;
  let v = array<vec3f, 3>(
    vert_attribs[verts[0]].pos.xyz,
    vert_attribs[verts[1]].pos.xyz,
    vert_attribs[verts[2]].pos.xyz
  );

  let norms = array<vec3f, 3>(
    vert_attribs[verts[0]].normal.xyz,
    vert_attribs[verts[1]].normal.xyz,
    vert_attribs[verts[2]].normal.xyz
  );

  let e0 = v[1] - v[0];
  let e1 = v[2] - v[0];
  let n = cross(e0, e1);
  let omega_dot_n = dot(ray.direction, n);

  if (abs(omega_dot_n) > 1e-8) {
    let origin_to_v0 = v[0] - ray.origin;
    let t = dot(origin_to_v0, n)/omega_dot_n;
    if (t <= ray.tmax && t >= ray.tmin) {
      let partial = cross(origin_to_v0, ray.direction);
      let beta = dot(partial, e1) / omega_dot_n;
      if (beta >= 0) {
        let gamma = -dot(partial, e0) / omega_dot_n;
        if (gamma >= 0 && (beta + gamma) <= 1) {
          let matIndex = vert_indices[i].w;
          if (matIndex >= arrayLength(&materials)) {
            (*hit).color_amb = vec3f(1.0);
          } else {
            let material = materials[matIndex];
            (*hit).color_diff = material.diffuse.rgb;
            (*hit).color_amb = material.emission.rgb;
          }

          (*hit).hit = true;
          (*hit).distance = t;
          (*hit).position = ray.origin + t * ray.direction;
          let alpha = 1.0 - (beta + gamma);
          (*hit).normal = normalize(alpha * norms[0] + beta * norms[1] + gamma * norms[2]);
          (*hit).shader = uniforms_int.shader;
        }
      }
    }
  }

  return (*hit).hit;
}



fn sample_directional_light(p: vec3f) -> Light {
  let Le = vec3f(3.14159);
  let dist = 1e9;
  let light_direction = normalize(-dir_light_dir);
  return Light(Le, light_direction, dist);
}

fn sample_env_light(p: vec3f, normal: vec3f, seed: ptr<function, u32>) -> Light {
  let sample_cos_theta = sqrt(1f - rnd(seed));
  let sample_phi = 2 * pi * rnd(seed);
  let sample_sin_theta = sqrt(1f - sample_cos_theta * sample_cos_theta);
  let dir = rotate_to_normal(
    normal,
    spherical_direction(sample_sin_theta, sample_cos_theta, sample_phi)
  );

  let Le = sample_skybox(Ray(p, dir, 1e-1, 1e9)) * 5;
  return Light(Le, dir, 1);
}

fn spherical_direction(sin_theta: f32, cos_theta: f32, phi: f32) -> vec3f {
  return vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

fn rotate_to_normal(n: vec3f, v: vec3f) -> vec3f {
  let s = sign(n.z + 1e-16f);
  let a = -1f/(1f + abs(n.z));
  let b = n.x * n.y * a;
  return vec3f(1.0f + n.x*n.x*a, b, -s*n.x)*v.x 
       + vec3f(s*b, s*(1.0f + n.y*n.y*a), -n.y)*v.y 
       + n*v.z;
}

fn check_shadow(pos: vec3f, lightdir: vec3f, lightdist: f32) -> bool {
  var lightray = Ray(pos, lightdir, 10e-4, lightdist - 10e-4);
  var lighthit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0),
                          vec3f(0.0), vec3f(0.0), 1.0, 1.0, 0, vec2f(0.0));
  return int_scene(&lightray, &lighthit);
}

fn lambert(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  var Lr = (*hit).color_amb / pi;
  let light = sample_env_light((*hit).position, (*hit).normal, seed);

  if (!check_shadow((*hit).position, light.wi, light.dist)) {
    Lr += ((*hit).color_diff / pi) * light.Li * max(dot((*hit).normal, light.wi), 0.0);
  }

  let dirLight = sample_directional_light((*hit).position);
  if (!check_shadow((*hit).position, dirLight.wi, dirLight.dist)) {
    Lr += ((*hit).color_diff / pi) * dirLight.Li * max(dot((*hit).normal, dirLight.wi), 0.0);
  }

  return Lr;
}

fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  let wo = normalize((*r).origin - (*hit).position);
  var Lr = (*hit).color_amb / pi;

  let light = sample_directional_light((*hit).position);
  if (!check_shadow((*hit).position, light.wi, light.dist)) {
    let wr = reflect(light.wi, (*hit).normal);
    Lr += light.Li * dot(light.wi, (*hit).normal) *
          (
            ((*hit).color_diff / pi) + 
            (
              (*hit).color_specular * ((*hit).shine + 2) * 0.15915494309 *
              pow(max(dot(wo, wr), 0.0), (*hit).shine)
            )
          );
  }

  return Lr;
}

fn shade_transparent(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  (*r).origin = (*hit).position;
  (*r).tmin = 1e-2;
  (*r).tmax = 1e6;
  (*hit).hit = false;

  let n = (*hit).refractive_ratio;
  let cos_in = -dot((*r).direction, (*hit).normal);
  let sin_sq_in = 1.0 - cos_in*cos_in;
  let sin_sq_out = (n * n) * sin_sq_in;
  let cos_sq_out = 1.0 - sin_sq_out;
  var R = fresnel_R(cos_in, sqrt(max(cos_sq_out, 0.0)), n);

  if (cos_sq_out < 0) {
    // Total internal reflection
    R = 1f;
  }

  let sample = rnd(seed);
  if (sample <= R) {
    (*r).direction = reflect((*r).direction, (*hit).normal);
    return vec3f(0);
  }

  let cos_out = sqrt(max(cos_sq_out, 0.0));
  (*r).direction = (n * (*r).direction) + (n * cos_in - cos_out) * (*hit).normal;
  return vec3f(0);
}

fn fresnel_R(cos_in: f32, cos_out: f32, index: f32) -> f32 {
  let r_perp = abs(index * (cos_in - (cos_out / index)) / (index * cos_in + cos_out));
  let r_parallel = abs(index * ((cos_in/index) - cos_out) / (cos_in + index * cos_out));
  let r_perp_sq = r_perp * r_perp;
  let r_par_sq = r_parallel * r_parallel;
  let R = 0.5 * (r_perp_sq + r_par_sq);
  return R;
}

fn shade_refract(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
  (*r).origin = (*hit).position;
  (*r).tmin = 1e-2;
  (*r).tmax = 1e6;
  (*hit).hit = false;

  let n = (*hit).refractive_ratio;
  let cos_in = dot((*r).direction, (*hit).normal);
  let sin_sq_in = 1.0 - cos_in*cos_in;
  let sin_sq_out = (n*n)*sin_sq_in;
  let cos_sq_out = 1.0 - sin_sq_out;

  if (cos_sq_out < 0) {
    (*r).direction = reflect((*r).direction, (*hit).normal);
    return vec3f(1);
  }

  (*r).direction = (n * (*r).direction) - (n*cos_in + sqrt(cos_sq_out))*(*hit).normal;
  return vec3f(0);
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  switch (*hit).shader {
    case shader_matte:       { return lambert(r, hit, seed); }
    case shader_reflect: {
      (*hit).hit = false;
      (*r).origin = (*hit).position;
      (*r).direction = reflect((*r).direction, (*hit).normal);
      (*r).tmin = 1e-2;
      (*r).tmax = 1e10;
      return vec3f(0.0);
    }
    case shader_refract:     { return shade_refract(r, hit); }
    case shader_phong:       { return phong(r, hit, seed); }
    case shader_glossy:      { return phong(r, hit, seed) + shade_refract(r, hit); }
    case shader_transparent: { return shade_transparent(r, hit, seed); }
    default:                 { return (*hit).color_diff + (*hit).color_amb; }
  }
}



fn texture_linear(texture: texture_2d<f32>, texcoords: vec2f) -> vec4f {
  let res = textureDimensions(texture);
  let st = texcoords - floor(texcoords);
  let ab = st * vec2f(res);

  let x1: u32 = u32(ab.x);
  let x2: u32 = u32(ab.x + 1);
  let y1: u32 = u32(ab.y);
  let y2: u32 = u32(ab.y + 1);
  let x1_f = f32(x1);
  let x2_f = f32(x2);
  let y1_f = f32(y1);
  let y2_f = f32(y2);

  let UV11 = vec2u(x1, y1);
  let UV12 = vec2u(x1, y2);
  let UV22 = vec2u(x2, y2);
  let UV21 = vec2u(x2, y1);

  var texcolor = vec4f(0.0);
  let denom = f32(((x2 - x1) * (y2 - y1)));
  texcolor += textureLoad(texture, UV11, 0) * ((x2_f - ab.x)*(y2_f - ab.y)) / denom;
  texcolor += textureLoad(texture, UV12, 0) * ((x2_f - ab.x)*(ab.y - y1_f)) / denom;
  texcolor += textureLoad(texture, UV21, 0) * ((ab.x - x1_f)*(y2_f - ab.y)) / denom;
  texcolor += textureLoad(texture, UV22, 0) * ((ab.x - x1_f)*(ab.y - y1_f)) / denom;
  return texcolor;
}

@fragment
fn main_fs(@builtin(position) fragcoord: vec4f, @location(0) coords: vec2f) -> FSOut {
  let launch_idx = u32(fragcoord.y) * uniforms_int.canvas_width + u32(fragcoord.x);
  var t = tea(launch_idx, uniforms_int.frame_num);

  let jitter = vec2f(rnd(&t), rnd(&t)) / f32(uniforms_int.canvas_height);
  const max_depth = 10;
  var result = vec3f(0.0);

  var hit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0),
                    vec3f(0.0), vec3f(0.0), 1.0, 1.0, 0, vec2f(0.0));
  var ipcoords = vec2f(coords.x * uniforms_f.aspect * 0.5, coords.y * 0.5);
  var r = get_camera_ray(ipcoords + jitter); 

  for (var i = 0; i < max_depth; i++) {
    if (int_scene(&r, &hit)) {
      result += shade(&r, &hit, &t);
      if (hit.hit) {
        break;
      }
      if (dot(result, result) >= 0.99) {
        break;
      }
    } else {
      r.tmax = 1e9;
      let bg_texture_sample = sample_skybox(r);
      result += bg_texture_sample * holdout_occlusion(r, &t);
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
