
use super::recommendation::Recommendations;

pub trait Policy {
    fn evaluate(&self) -> Recommendations;
    // fn act(&self, state: Vec<f64>) -> Vec<f64>;
    // fn update(&mut self, state: Vec<f64>, action: Vec<f64>, reward: f64, next_state: Vec<f64>);
}

// pub struct RandomPolicy {}

// impl Policy for RandomPolicy {
//     fn evaluate(&self) -> Recommendations {
//         // we want to return a random recommendation
//         // first we sample from the set of possible actions
//         let possible_actions = vec![Action::Hold, Action::Open {
//             order: Order::Market, position: Position::Long(PositionSize {
//                  symbol: "AAPL".to_string(), size: 1.0, price: None }) }];
//         // then we return a recommendation with that action and a random confidence



//         let recommendations = vec![
//             Recommendation {
//                 action: Action::Hold,
//                 confidence: 0.5,
//             },
//             Recommendation {
//                 action: Action::Open {
//                     order: Order::Market,
//                     position: Position::Long(PositionSize {
//                         symbol: "AAPL".to_string(),
//                         size: 1.0,
//                         price: None,
//                     }),
//                 }
//             }
//         ];
    
//         let mut recommendations = vec![
//             Recommendation {
//                 action: Action::Hold,
//                 confidence: 0.5,
//             },
//             Recommendation {
//                 action: Action::Open {
//                     position: Position::Long(PositionSize {
//                         symbol: "AAPL".to_string(),
//                         size: 1.0,
//                         price: 100.0,
//                     }),
//                     order: Order::Market,
//                 },
//                 confidence: 0.25,
//             },
//         ];
//         Recommendations { recommendations }
//     }
// }

pub struct BuyAndHoldPolicy {}



//
