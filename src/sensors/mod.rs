// No topo de cada arquivo, adicione:
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

pub mod hardware;
pub mod camera;
pub mod audio;
pub mod sensor_control;
pub mod vision_stream;
pub use sensor_control::SensorFlags;
use crate::brain_zones::RegionType;