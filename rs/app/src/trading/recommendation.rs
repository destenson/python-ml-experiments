
#[derive(Debug, Serialize, Deserialize, Clone)]
enum Order {
    Market { price: Option<f64>, size: f64 },
    Limit { price: f64, size: f64 },
    Stop { price: f64, size: f64 },
    StopLimit { stop: f64, limit: f64, size: f64 },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PositionSize {
    pub symbol: String,
    pub size: f64,
    pub price: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum Positions {
    Long(PositionSize),
    Short(PositionSize),
    Flat(PositionSize),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum Action {
    // Actions that can be recommended
    Open{position: Position, order: Order},
    Close{position: Position, order: Order},
    Hold,
    ResearchHistory{symbol: String, start: i64, end: i64},
    ResearchNews{symbol: String, start: i64, end: i64},
    ResearchFundamentals{symbol: String, start: i64, end: i64},
    ResearchTechnicals{symbol: String, start: i64, end: i64},
    ResearchSentiment{symbol: String, start: i64, end: i64},
    ResearchEconomic{symbol: String, start: i64, end: i64},
    ResearchMacro{symbol: String, start: i64, end: i64},
    ResearchMarket{symbol: String, start: i64, end: i64},
    ResearchOptions{symbol: String, start: i64, end: i64},
    ResearchFutures{symbol: String, start: i64, end: i64},
    ResearchForex{symbol: String, start: i64, end: i64},
    ResearchCrypto{symbol: String, start: i64, end: i64},
    ResearchCommodities{symbol: String, start: i64, end: i64},
    ResearchBonds{symbol: String, start: i64, end: i64},
    ResearchStocks{symbol: String, start: i64, end: i64},
    ResearchETFs{symbol: String, start: i64, end: i64},
    ResearchMutualFunds{symbol: String, start: i64, end: i64},
    ResearchIndices{symbol: String, start: i64, end: i64},
    ResearchSectors{symbol: String, start: i64, end: i64},
    ResearchIndustries{symbol: String, start: i64, end: i64},
    ResearchThemes{symbol: String, start: i64, end: i64},
    ResearchEvents{symbol: String, start: i64, end: i64},
    ResearchEarnings{symbol: String, start: i64, end: i64},
    ResearchDividends{symbol: String, start: i64, end: i64},
    ResearchSplits{symbol: String, start: i64, end: i64},
    ResearchIPOs{symbol: String, start: i64, end: i64},
    ResearchBuybacks{symbol: String, start: i64, end: i64},
}

// recommendation for an action after a policy evaluation
#[derive(Debug, Serialize, Deserialize, Clone)]
struct Recommendation {
    action: Action,
    confidence: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Recommendations {
    recommendations: Vec<Recommendation>,
}

pub struct Portfolio {
    pub positions: Vec<Position>,
    // pub cash: f64,
    // pub positions: HashMap<String, Position>,
    // pub orders: Vec<Order>,
}
