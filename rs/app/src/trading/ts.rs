
pub trait TimeSeriesModel {
    fn fit(&mut self, x: Vec<f64>, y: Vec<f64>);
}

pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
}

pub struct FourierSeasonality {
    pub period: usize,
    pub harmonics: usize,
}

pub struct ExponentialSmoothing {
    pub alpha: f64,
    pub level: f64,
}

pub struct ARIMA {
    pub p: usize,
    pub d: usize,
    pub q: usize,
}

pub struct GARCH {
    pub p: usize,
    pub q: usize,
}

pub struct TimeSeries {
    pub model: Box<dyn TimeSeriesModel>,
}

impl TimeSeries {
    pub fn new(model: Box<dyn TimeSeriesModel>) -> Self {
        TimeSeries { model }
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.model.fit(x, y);
    }
}

impl TimeSeriesModel for LinearRegression {
    fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_x_squared = x.iter().map(|x| x.powi(2)).sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(x, y)| x * y).sum::<f64>();

        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x.powi(2));
        self.intercept = (sum_y - self.slope * sum_x) / n;
    }
}

impl TimeSeriesModel for FourierSeasonality {
    fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        unimplemented!();
    }
}

impl TimeSeriesModel for ExponentialSmoothing {
    fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.level = y[0];
        for i in 1..y.len() {
            self.level = self.alpha * y[i] + (1.0 - self.alpha) * self.level;
        }
    }
}

impl TimeSeriesModel for ARIMA {
    fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        unimplemented!();
    }
}

impl TimeSeriesModel for GARCH {
    fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        unimplemented!();
    }
}

// #[test]
// fn test_main() {
//     let mut ts = TimeSeries::new(Box::new(LinearRegression { slope: 0.0, intercept: 0.0 }));
//     ts.fit(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]);
//     println!("Slope: {}, Intercept: {}", ts.model.slope, ts.model.intercept);

//     let mut ts = TimeSeries::new(Box::new(ExponentialSmoothing { alpha: 0.5, level: 0.0 }));
//     ts.fit(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]);
//     println!("Level: {}", ts.model.level);
// }
