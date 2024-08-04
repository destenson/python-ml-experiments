
pub mod trading;
pub mod glm;
pub mod vi;
pub mod data;

use ndarray::Array;
use ndarray_rand::rand_distr::StandardNormal;

use vi::VariationalInference;

fn main_() {
    println!("Hello, world!");
}

fn main() {
    let input_dim = 784; // Example input dimension (e.g., MNIST)
    let latent_dim = 2;
    // let vi = VariationalInference::new(input_dim, latent_dim);

    // // Example data
    // let x = Array::random((1, input_dim), StandardNormal);

    // // Forward pass
    // let (x_hat, mean, std, z, loss) = vi.forward(&x);
    // println!("Reconstructed: {:?}", x_hat);
    // println!("Mean: {:?}", mean);
    // println!("Std: {:?}", std);
    // println!("Latent: {:?}", z);
    // println!("Loss: {:?}", loss);
}
