// extern crate tensorflow;
// extern crate ndarray;
// extern crate ndarray_rand;
// extern crate num_traits;

use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor};
use tensorflow::train::Optimizer;
use tensorflow::ops::{self, Placeholder};
use ndarray::{Array, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
// use num_traits::Float;

pub struct VariationalInference {
    encoder: Graph,
    decoder: Graph,
    session: Session,
}

// impl VariationalInference {
//     pub fn new(input_dim: usize, latent_dim: usize) -> Self {
//         let mut encoder = Graph::new();
//         let mut decoder = Graph::new();
//         let session = Session::new(&SessionOptions::new(), &encoder).unwrap();

//         // Define the encoder
//         let input = Placeholder::new();
//         //(&mut encoder, &tensorflow::DataType::Float);
//         let weights = Tensor::new(&[input_dim as u64, latent_dim as u64 * 2])
//             .with_values(&Array::random((input_dim, latent_dim * 2), StandardNormal).into_raw_vec())
//             .unwrap();
//         let biases = Tensor::new(&[latent_dim as u64 * 2])
//             .with_values(&Array::zeros(latent_dim * 2).into_raw_vec())
//             .unwrap();
//         let linear = ops::mat_mul(&mut encoder, input.clone(), weights, None);
//         let linear = ops::add(&mut encoder, linear, biases);

//         // Define the decoder
//         let latent_input = Placeholder::new(&mut decoder, &tensorflow::DataType::Float);
//         let dec_weights = Tensor::new(&[latent_dim as u64, input_dim as u64])
//             .with_values(&Array::random((latent_dim, input_dim), StandardNormal).into_raw_vec())
//             .unwrap();
//         let dec_biases = Tensor::new(&[input_dim as u64])
//             .with_values(&Array::zeros(input_dim).into_raw_vec())
//             .unwrap();
//         let dec_linear = ops::mat_mul(&mut decoder, latent_input.clone(), dec_weights, None);
//         let dec_linear = ops::add(&mut decoder, dec_linear, dec_biases);
//         let output = ops::sigmoid(&mut decoder, dec_linear);

//         Self { encoder, decoder, session }
//     }

//     fn gaussian_parameters(&self, h: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
//         let mean = h.slice(s![.., 0..h.shape()[1] / 2]).to_owned();
//         let std = h.slice(s![.., h.shape()[1] / 2..]).mapv(|x| x.exp() / 2.0).to_owned();
//         (mean, std)
//     }

//     fn sample_z(&self, mean: &Array2<f32>, std: &Array2<f32>) -> Array2<f32> {
//         let eps = Array::random(mean.raw_dim(), StandardNormal);
//         mean + std * eps
//     }

//     fn kl_divergence(&self, mean: &Array2<f32>, std: &Array2<f32>) -> f32 {
//         let kl = (1.0 + std.mapv(|x| x.ln()) - mean.mapv(|x| x.powi(2)) - std.mapv(|x| x.powi(2))).sum();
//         -0.5 * kl
//     }

//     fn elbo_loss(&self, x: &Array2<f32>, x_hat: &Array2<f32>, mean: &Array2<f32>, std: &Array2<f32>) -> f32 {
//         let reconstruction_loss = (x - x_hat).mapv(|x| x.powi(2)).sum();
//         let kl_loss = self.kl_divergence(mean, std);
//         reconstruction_loss + kl_loss
//     }

//     pub fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, f32) {
//         // Encode
//         let mut run_args = SessionRunArgs::new();
//         let input_tensor = Tensor::new(&[x.shape()[0] as u64, x.shape()[1] as u64])
//             .with_values(x.as_slice().unwrap())
//             .unwrap();
//         run_args.add_feed(&self.encoder.operation_by_name_required("Placeholder").unwrap(), 0, input_tensor);
//         let encoded = run_args.request_fetch(&self.encoder.operation_by_name_required("Add").unwrap(), 0);
//         self.session.run(&mut run_args).unwrap();
//         let h: Tensor<f32> = run_args.fetch(encoded).unwrap().into();

//         // Gaussian parameters
//         let h = Array::from_shape_vec((h.dims()[0] as usize, h.dims()[1] as usize), h.to_vec()).unwrap();
//         let (mean, std) = self.gaussian_parameters(&h);

//         // Sample z
//         let z = self.sample_z(&mean, &std);

//         // Decode
//         let mut run_args = SessionRunArgs::new();
//         let z_tensor = Tensor::new(&[z.shape()[0] as u64, z.shape()[1] as u64])
//             .with_values(z.as_slice().unwrap())
//             .unwrap();
//         run_args.add_feed(&self.decoder.operation_by_name_required("Placeholder").unwrap(), 0, &z_tensor);
//         let decoded = run_args.request_fetch(&self.decoder.operation_by_name_required("Sigmoid").unwrap(), 0);
//         self.session.run(&mut run_args).unwrap();
//         let x_hat: Tensor<f32> = run_args.fetch(decoded).unwrap().into();

//         let x_hat = Array::from_shape_vec((x_hat.dims()[0] as usize, x_hat.dims()[1] as usize), x_hat.to_vec()).unwrap();
//         let loss = self.elbo_loss(x, &x_hat, &mean, &std);

//         (x_hat, mean, std, z, loss)
//     }
// }
