
// TODO: use crates instead of writing my own versions of these

pub enum Family {
    Gaussian,
    Poisson,
    Bernoulli,
    Binomial,
}

pub mod traits {

    pub trait LinkFunction {
        fn link(&self, x: f64) -> f64;
        fn inverse_link(&self, x: f64) -> f64;
    }
    
    pub trait GeneralizedLinearModel {
        fn fit(&mut self, x: Vec<f64>, y: Vec<f64>);
        fn predict(&self, x: Vec<f64>) -> Vec<f64>;
        fn spec(&self) -> super::GeneralizedLinearModelSpec;
    }
    
}

pub struct IdentityLink {}

impl traits::LinkFunction for IdentityLink {
    fn link(&self, x: f64) -> f64 {
        x
    }

    fn inverse_link(&self, x: f64) -> f64 {
        x
    }
}

pub struct GeneralizedLinearModelSpec {
    pub family: String,
    pub link: String,
}

pub struct GeneralizedLinearModel {
    pub family: Family,
    pub link: Box<dyn traits::LinkFunction>,
}

impl GeneralizedLinearModel {
    pub fn new(spec: GeneralizedLinearModelSpec) -> Self {
        let family = match spec.family.as_str() {
            "Gaussian" => Family::Gaussian,
            "Poisson" => Family::Poisson,
            "Bernoulli" => Family::Bernoulli,
            "Binomial" => Family::Binomial,
            _ => Family::Gaussian,
        };

        let link = match spec.link.as_str() {
            "Identity" => Box::new(IdentityLink {}),
            _ => Box::new(IdentityLink {}),
        };

        GeneralizedLinearModel { family, link }
    }
}

impl GeneralizedLinearModel {
    pub fn linear_response(&self, input: &[f64]) -> f64 {
        input.iter().fold(0.0, |acc, x| acc + x*self.link.link(*x))
    }
}

impl GeneralizedLinearModel {
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        // fit the model
    }

    pub fn predict(&self, x: Vec<f64>) -> Vec<f64> {
        // predict the model
        vec![]
    }
}
