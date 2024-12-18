struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) coords: vec2f
}

struct UniformsF { // float values
  aspect: f32,
  gamma: f32,
}
struct UniformsInt { // unsigned int values
  canvas_width: u32,
  canvas_height: u32,
  frame_num: u32
}

// Axis-Aligned Bounding box (box between the points min and max)
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

@group(0) @binding(0) var<uniform> uniforms_f: UniformsF;
@group(0) @binding(1) var<uniform> uniforms_int: UniformsInt;
@group(0) @binding(3) var<uniform> aabb: AABB;

@group(0) @binding(4) var<storage> vert_attribs: array<VertexAttrib>;
@group(0) @binding(6) var<storage> materials: array<Material>; // colors
@group(0) @binding(7) var<storage> vert_indices: array<vec4u>;
@group(0) @binding(8) var<storage> treeIds: array<u32>;
@group(0) @binding(9) var<storage> bspTree: array<vec4u>;
@group(0) @binding(10) var<storage> bspPlanes: array<f32>;

@group(0) @binding(11) var<storage> light_indices: array<u32>;
@group(0) @binding(12) var renderTexture: texture_2d<f32>;

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex: u32) -> VSOut {
  const pos = array<vec2f, 4>(vec2f(-1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(1, -1));
  var vsOut: VSOut;
  vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
  vsOut.coords = pos[VertexIndex];
  return vsOut;
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
  color_amb: vec3f, 
  color_diff: vec3f, 
  color_specular: vec3f,
  shine: f32, 
  refractive_ratio: f32, 
  shader: u32,
  texcoords: vec2f,
  emit: bool,
  factor: vec3f,
  extinction: vec3f, // NEW: Extinction coefficient (σt)
}

// shader type names:
const shader_matte: u32 = 1;
const shader_reflect: u32 = 2;
const shader_refract: u32 = 3;
const shader_phong: u32 = 4;
const shader_glossy: u32 = 5;
const shader_transparent: u32 = 6;

// Refraction indices
const refr_exit = 1.5;   // inside medium
const refr_enter= 0.667; // entering medium

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
  const LCG_A = 1977654935u; 
  *prev = (LCG_A * (*prev)) & 0x7FFFFFFF;
  return *prev; 
}

// generates pseudo random f32 in range [0, 1)
fn rnd(prev: ptr<function, u32>) -> f32 {
  return f32(mcg31(prev)) / f32(0x80000000);
}

const tex_scale = 0.02;
const e = vec3f(277, 275.0, -570.0);
const d = 1.0f;
const u = vec3f(0, 1, 0);
const p = vec3f(277.0, 275.0, 0.0);
const dir_light_dir = vec3f(-0.3, -0.1, -0.8);

struct Onb {
  tangent: vec3f,
  binormal: vec3f,
  normal: vec3f,
};

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

// Fresnel reflectance computation
fn fresnel_R(cos_theta_i: f32, cos_theta_t: f32, eta: f32) -> f32 {
  if(cos_theta_t < 0.0) {
    return 1.0;
  }
  let Rs = ((eta*cos_theta_i - cos_theta_t) / (eta*cos_theta_i + cos_theta_t));
  let Rp = ((cos_theta_i - eta*cos_theta_t) / (cos_theta_i + eta*cos_theta_t));
  return 0.5 * (Rs*Rs + Rp*Rp);
}

fn int_scene(ray: ptr<function, Ray>,  hit: ptr<function, HitInfo>) -> bool {
  const sphere_1_c = vec3f(420.0, 90.0, 370.0);
  const sphere_2_c = vec3f(130.0, 90.0, 250.0);
  const sphere_r = 90f;

  if(int_sphere(*ray, hit, sphere_1_c, sphere_r, shader_reflect)){
    (*ray).tmax = (*hit).distance;
  }

  // Glass sphere on the right - use transparent shader
  if(int_sphere(*ray, hit, sphere_2_c, sphere_r, shader_transparent)){
    (*ray).tmax = (*hit).distance;
  }

  if(int_aabb(ray)){
    if(int_trimesh(ray, hit)){
      (*ray).tmax = (*hit).distance;
    }
  }

  return (*hit).hit;
}

fn int_sphere(ray: Ray, hit: ptr<function, HitInfo>, center: vec3f, radius: f32, shade_mode: u32) -> bool {
  const sphere_color = vec3f(0,0,0);
  
  let dist_from_origin = ray.origin - center;
  let half_b = dot(dist_from_origin, ray.direction);
  let c = dot(dist_from_origin, dist_from_origin) - (radius * radius);
  let desc = (half_b * half_b) - c;
  
  if(desc < 0){
    return (*hit).hit;
  }

  let t = -half_b;
  if(desc <= 1e-4){
    if(ray.tmax >= t && ray.tmin <= t){
      (*hit).color_diff = sphere_color * 0.9;
      (*hit).color_amb = sphere_color * 0.1;
      (*hit).color_specular = vec3f(0.1);
      (*hit).shine = 42;
      (*hit).hit = true;
      (*hit).distance = t;
      let pos = ray.origin + t * ray.direction;
      (*hit).position = pos;
      (*hit).normal = normalize(pos - center);
      (*hit).shader = shade_mode;
      (*hit).refractive_ratio = 1.0;
      (*hit).extinction = vec3f(0.0); // no extinction for non-transparent
    }

  } else { 
    let sqrt_desc = sqrt(desc);
    let t1 = t - sqrt_desc;
    let t2 = t + sqrt_desc;
    if(ray.tmax >= t1 && ray.tmin <= t1){ 
      (*hit).distance = t1;
      (*hit).hit = true;
    } else if (ray.tmax >= t2 && ray.tmin <= t2){
      (*hit).distance = t2;
      (*hit).hit = true;
    }

    if((*hit).hit){
      let pos = ray.origin + ((*hit).distance * ray.direction);
      (*hit).position = pos;
      (*hit).color_amb = sphere_color * 0.1;
      (*hit).color_diff = sphere_color * 0.9;
      (*hit).color_specular = vec3f(0.1);
      (*hit).shine = 42;
      (*hit).normal = normalize((*hit).position - center);
      (*hit).shader = shade_mode;
      (*hit).refractive_ratio = refr_enter;

      if (dot(ray.direction, (*hit).normal) > 0) {
          (*hit).normal = -(*hit).normal;
          (*hit).refractive_ratio = refr_exit;
      }

      // NEW: If this is the transparent sphere, set a non-zero extinction
      if(shade_mode == shader_transparent) {
        // Choose an RGB extinction coefficient to give the sphere a tinted absorption
        (*hit).extinction = vec3f(0.1, 0.05, 0.2); // example values
      } else {
        (*hit).extinction = vec3f(0.0);
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
  if(tmin > tmax || tmin > r.tmax || tmax < r.tmin){
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

fn int_trimesh(ray: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool{
  var branch_level = 0u;
  var near_node = 0u;
  var far_node = 0u;
  var t = 0.0f;
  var node = 0u;
  for(var i = 0u; i <= MAX_LEVEL; i ++){
    let tree_node = bspTree[node];
    let node_axis_leaf = tree_node.x &3u;
    if(node_axis_leaf == BSP_LEAF){
      let node_count = tree_node.x >> 2u;
      let node_id = tree_node.y;
      var found = false;
      for(var j = 0u; j < node_count; j++){
        let obj_idx = treeIds[node_id + j];
        if(int_triangle(*ray, hit, obj_idx)){
          (*ray).tmax = (*hit).distance;
          found = true;
        }
      }
      if(found){ return true;}
      else if(0u == branch_level){
        return false;
      }
      else {
        branch_level --;
        i = branch_node[branch_level].x;
        node = branch_node[branch_level].y;
        (*ray).tmin = branch_ray[branch_level].x;
        (*ray).tmax = branch_ray[branch_level].y;
        continue;
      }
    }
    let axis_direction = (*ray).direction[node_axis_leaf];
    let axis_origin = (*ray).origin[node_axis_leaf];
    if(axis_direction >= 0.0){
      near_node = tree_node.z; 
      far_node = tree_node.w;
    } else {
      near_node = tree_node.w; 
      far_node = tree_node.z; 
    }
    let node_plane = bspPlanes[node];
    var denom = axis_direction;
    if(abs(denom) < 1e-8){
      if (denom < 0) {
        denom = -1e-8;
      } else {
        denom = 1e-8;
      }
    }
    t = (node_plane - axis_origin) / denom;
    if(t >= (*ray).tmax){
      node = near_node;
    } else if(t <= (*ray).tmin) {
      node = far_node;
    } else {
      branch_node[branch_level].x = i;
      branch_node[branch_level].y = far_node;
      branch_ray[branch_level].x = t;
      branch_ray[branch_level].y = (*ray).tmax;
      branch_level ++;
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
    vert_attribs[verts[2]].normal.xyz,
  );

  let e0 = v[1] - v[0];
  let e1 = v[2] - v[0];
  let n = cross(e0, e1);
  let omega_dot_n = dot(ray.direction, n);

  if(abs(omega_dot_n) > 1e-8){
    let origin_to_v0 = v[0] - ray.origin;
    let t = dot(origin_to_v0, n)/omega_dot_n;
    if(t <= ray.tmax && ray.tmin <= t){
      let partial = cross(origin_to_v0, ray.direction);
      let beta = dot(partial, e1) / omega_dot_n;
      if(beta >= 0){
        let gamma = -dot(partial, e0) / omega_dot_n;
        if(gamma >= 0 && (beta + gamma) <= 1){
          let matIndex = vert_indices[i].w;
          let material = materials[matIndex];
          (*hit).color_diff = material.diffuse.rgb;
          (*hit).color_amb = material.emission.rgb;
          if(length(hit.color_amb) > 1e-1){
            hit.emit = true;
          } else {
            hit.emit = false;
          }
          (*hit).hit = true;
          (*hit).distance = t;
          (*hit).position = ray.origin + t * ray.direction;
          let alpha = 1.0 - (beta + gamma);
          (*hit).normal = normalize(alpha * norms[0] + beta * norms[1] + gamma*norms[2]);
          (*hit).shader = shader_matte;
          (*hit).extinction = vec3f(0.0);
        }
      }
    }
  }

  return (*hit).hit;
}

struct Light {
  Li: vec3f,
  wi: vec3f,
  dist: f32,
}

// Sample area light
fn sample_trimesh_light(p: vec3f, seed: ptr<function, u32>) -> Light {
  let numTriangles = arrayLength(&light_indices);
  var totalArea = 0f;
  for(var i = 0u; i < numTriangles; i ++){
    let idx = light_indices[i];
    let vs = vert_indices[idx].xyz;
    let q = array<vec3f, 3>(
      vert_attribs[vs[0]].pos.xyz,
      vert_attribs[vs[1]].pos.xyz,
      vert_attribs[vs[2]].pos.xyz
    );
    let area = length(cross(q[1]-q[0], q[2]-q[0]));
    totalArea += abs(area);
  }
  let sampled_triangle_area = rnd(seed) * totalArea;
  var sample_idx = 0u;
  var cumulArea = 0f;
  for(var i = 0u; i < numTriangles; i ++){
    let idx = light_indices[i];
    let vs = vert_indices[idx].xyz;
    let q = array<vec3f, 3>(
      vert_attribs[vs[0]].pos.xyz,
      vert_attribs[vs[1]].pos.xyz,
      vert_attribs[vs[2]].pos.xyz
    );
    let area = length(cross(q[1]-q[0], q[2]-q[0]));
    cumulArea += area;
    if(cumulArea >= sampled_triangle_area){
      sample_idx = i;
      break;
    }
  }
  let index = light_indices[sample_idx];
  let verts = vert_indices[index].xyz;
  let q = array<vec3f, 3>(
    vert_attribs[verts[0]].pos.xyz,
    vert_attribs[verts[1]].pos.xyz,
    vert_attribs[verts[2]].pos.xyz
  );
  let norms = array<vec3f, 3>(
    vert_attribs[verts[0]].normal.xyz,
    vert_attribs[verts[1]].normal.xyz,
    vert_attribs[verts[2]].normal.xyz,
  );
  let r1 = rnd(seed);
  let r2 = rnd(seed);
  let alpha = 1f - sqrt(r1);
  let beta = (1f - r2) * sqrt(r1);
  let gamma = r2 * sqrt(r1);

  let point = alpha*q[0] + beta*q[1] + gamma*q[2];
  let norm = alpha*norms[0] + beta*norms[1] + gamma*norms[2];

  let e0 = q[1] - q[0];
  let e1 = q[2] - q[0];
  let areaCross = cross(e0, e1);
  let area = length(areaCross)*0.5;

  var Le = materials[vert_indices[index].w].emission.rgb;
  let dist = distance(p, point);
  let wi = normalize(point - p);
  let Li = dot(-wi, norm)* Le * area * pow(1/dist, 2);
  return Light(Li, wi, dist);
}

fn check_shadow(pos: vec3f, lightdir: vec3f, lightdist: f32) -> bool {
  var lightray =  Ray(pos, lightdir, 10e-4, lightdist-10e-4);
  var lighthit = HitInfo(false, 0.0, vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), 1.0, 1.0, 0, vec2f(0.0), false, vec3f(1.0), vec3f(0.0));
  return int_scene(&lightray, &lighthit);
}

// The following shading functions remain mostly unchanged

fn lambert(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  hit.factor = hit.factor * hit.color_diff;
  let Pd = (hit.factor.x + hit.factor.y + hit.factor.z) * 0.333;
  let decision = rnd(seed);
  if(decision < Pd){
    hit.hit = false;
    hit.factor = hit.factor / Pd;
    r.origin = hit.position;
    let sample_cos_theta = sqrt(1f - rnd(seed));
    let sample_phi = 2 * 3.14159 * rnd(seed);
    let sample_sin_theta = sqrt(1f - sample_cos_theta * sample_cos_theta);
    let sampledDir = rotate_to_normal(
      hit.normal,
      spherical_direction(sample_sin_theta, sample_cos_theta, sample_phi)
    );
    r.direction = sampledDir;
    (*r).tmin = 1e-2;
    (*r).tmax = 1e10;
    return vec3f(0);
  }

  var Lr = ((*hit).color_amb / 3.14159);
  let light = sample_trimesh_light((*hit).position, seed);
  if(!check_shadow((*hit).position, light.wi, light.dist)){
    Lr += ((*hit).color_diff / (3.14159)) * light.Li * max(dot((*hit).normal, light.wi), 0.0);
  }
  return Lr;
}

fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  let wo = normalize((*r).origin - (*hit).position);
  var Lr = ((*hit).color_amb / 3.14159);

  let light = sample_trimesh_light((*hit).position, seed);
  if(!check_shadow((*hit).position, light.wi, light.dist)){
    let wr = reflect(light.wi, (*hit).normal);
    Lr += 
    light.Li * 
    dot(light.wi, (*hit).normal) *
    (
      ((*hit).color_diff / 3.14159) + 
      (
        (*hit).color_specular * 
        ((*hit).shine + 2) * 0.15915494309 *
        pow(max(dot(wo, wr), 0.0), (*hit).shine)
      )
    );
  }

  if(dot(Lr, Lr) < 0.5){
    return vec3f(0);
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
  let sin_sq_in = (1.0 - cos_in * cos_in);
  let sin_sq_out = (n * n) * sin_sq_in;
  let cos_sq_out = 1.0 - sin_sq_out;
  if(cos_sq_out < 0){
    (*r).direction = reflect((*r).direction, (*hit).normal);
    return vec3f(1);
  }
  (*r).direction = 
      (n * (*r).direction) - (n  * cos_in + sqrt(cos_sq_out)) * (*hit).normal;
  return vec3f(0);
}

// NEW: Extended transparent shader with absorption
fn shade_transparent(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  (*r).origin = (*hit).position;
  (*r).tmin = 1e-2;
  (*r).tmax = 1e6;
  (*hit).hit = false;

  let n = (*hit).refractive_ratio;
  let cos_in = -dot((*r).direction, (*hit).normal);

  let sin_sq_in = 1.0 - cos_in * cos_in;
  let sin_sq_out = (n * n) * sin_sq_in;
  let cos_sq_out = 1.0 - sin_sq_out;

  var R = 0.0;
  var cos_out = 0.0;
  if(cos_sq_out < 0){
    // total internal reflection
    R = 1.0;
  } else {
    cos_out = sqrt(cos_sq_out);
    R = fresnel_R(cos_in, cos_out, n);
  }

  // Check if we are inside the medium:
  // If refractive_ratio == refr_exit means currently inside
  let inside = ((*hit).refractive_ratio == refr_exit);
  if(inside) {
    // Apply Bouguer’s law
    let Tr = exp(-(*hit).extinction * (*hit).distance);
    let prob = (Tr.x + Tr.y + Tr.z) / 3.0;
    let sample = rnd(seed);
    if(sample > prob) {
      // Absorption (terminate path)
      return vec3f(0.0);
    }
    // Continue with transmission
    hit.factor = hit.factor * (Tr / prob);
  }

  let sample2 = rnd(seed);
  if(sample2 <= R){
    // reflection
    (*r).direction = reflect((*r).direction, (*hit).normal); 
    return vec3f(0);
  } else {
    // refraction
    (*r).direction = (n * (*r).direction) + (n * cos_in - cos_out)*(*hit).normal;
    return vec3f(0);
  }
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>, seed: ptr<function, u32>) -> vec3f {
  switch (*hit).shader {
    case shader_matte { return lambert(r, hit, seed);}
    case shader_reflect {
      (*hit).hit = false;
      (*r).origin = (*hit).position;
      (*r).direction = reflect((*r).direction, (*hit).normal);
      (*r).tmin = 1e-2;
      (*r).tmax = 1e10;
      return vec3f(0.0);
    } 
    case shader_refract {
      return shade_refract(r, hit);
    }
    case shader_phong {
      return phong(r, hit, seed);
    }
    case shader_glossy {
      return phong(r, hit, seed) + shade_refract(r, hit);
    }
    case shader_transparent {
      return shade_transparent(r, hit, seed);
    }
    default {return (*hit).color_diff + (*hit).color_amb;}
  }  
}

fn spherical_direction(sin_theta: f32, cos_theta: f32, phi: f32) -> vec3f {
  return vec3f(sin_theta * cos(phi), sin_theta*sin(phi), cos_theta);
}

fn rotate_to_normal(n: vec3f, v: vec3f) -> vec3f {
  let s = sign(n.z + 1e-16f);
  let a = -1f/(1f + abs(n.z));
  let b = n.x * n.y * a;
  return 
    vec3f(1.0f + n.x*n.x*a, b, -s*n.x)*v.x 
    + vec3f(s*b, s*(1.0f + n.y*n.y*a), -n.y)*v.y 
    + n*v.z;
}

struct FSOut {
  @location(0) frame: vec4f,
  @location(1) accum: vec4f,
}

@fragment
fn main_fs(@builtin(position) fragcoord: vec4f, @location(0) coords: vec2f) -> FSOut {
  let launch_idx = u32(fragcoord.y)*uniforms_int.canvas_width + u32(fragcoord.x);
  var t = tea(launch_idx, uniforms_int.frame_num); 
  let jitter = vec2f(rnd(&t), rnd(&t)) / f32(uniforms_int.canvas_height);

  const bgcolor = vec4f(0,0,0,1);
  const max_depth = 10;
  var result = vec3f(0.0);

  var hit = HitInfo(
    false, 0.0, vec3f(0.0), vec3f(0.0),
    vec3f(0.0), vec3f(0.0), vec3f(0.0),
    1.0, 1.0, 0, vec2f(0.0), false, vec3f(1.0), vec3f(0.0)
  );

  var ipcoords = vec2f((coords.x)*uniforms_f.aspect*0.5, (coords.y)*0.5);
  var r = get_camera_ray(ipcoords + jitter); 
  for(var i = 0; i < max_depth; i ++){
    if(int_scene(&r, &hit)){
      if(hit.emit){
        result += hit.color_amb * hit.factor;
      } else {
        result += shade(&r, &hit, &t);
      }
      if(hit.hit){
        break;
      }
      if(dot(result, result) >= 0.99){
        break;
      }
    } else {
      result += bgcolor.rgb;
      break;
    }
  }

  let curr_sum = textureLoad(renderTexture, vec2u(fragcoord.xy), 0).rgb
    * f32(uniforms_int.frame_num);
  let accum_color = (result + curr_sum)/ f32(uniforms_int.frame_num + 1u);
  var out: FSOut;
  out.frame = vec4f(pow(accum_color, vec3f(1.0 / uniforms_f.gamma)), 1.0);
  out.accum = vec4f(accum_color, 1.0);
  return out; 
}
