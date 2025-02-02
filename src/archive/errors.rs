/// # GridError
///
/// ## Description
/// `GridError` represents an error that can occur when creating an instance of
/// `Grid`.
///
#[derive(Debug)]
pub enum GridError {
    InvalidRange,
    InvalidNumPoints,
}

impl std::fmt::Display for GridError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            GridError::InvalidRange => {
                write!(
                    f,
                    "Invalid range: start_point must be less than end_point"
                )
            }
            GridError::InvalidNumPoints => {
                write!(f, "Invalid number of points: num_points must be greater than 1")
            }
        }
    }
}

impl std::error::Error for GridError {}

/// # ArithmeticError
///
/// ## Description
/// `ArithmeticError` represents an error that can occur when performing
/// arithmetic operations.
///
#[derive(Debug)]
pub enum ArithmeticError {
    DivisionByZero,
}

impl std::fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            ArithmeticError::DivisionByZero => {
                write!(f, "Division by zero: denominator is zero")
            }
        }
    }
}

impl std::error::Error for ArithmeticError {}

/// # FieldError
///
/// ## Description
/// `FieldError` represents an error that can occur when performing operations
/// on scalar and vector fields.
///
#[derive(Debug)]
pub enum FieldError {
    GridMismatch,
    OperationNotSupported,
    ArithmeticError(ArithmeticError),
}

impl std::fmt::Display for FieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            FieldError::GridMismatch => {
                write!(f, "Grid mismatch: invalid number of grid points")
            }
            FieldError::OperationNotSupported => {
                write!(f, "Operation not supported: operation is either invalid, or has not been implemented")
            }
            FieldError::ArithmeticError(ArithmeticError::DivisionByZero) => {
                write!(f, "Division by zero: denominator is zero")
            }
        }
    }
}

impl std::error::Error for FieldError {}
