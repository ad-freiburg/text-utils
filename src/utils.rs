use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use num::traits::{Num, NumAssignOps};

pub(crate) type Matrix<T> = Vec<Vec<T>>;

pub(crate) fn get_progress_bar(size: u64, hidden: bool) -> ProgressBar {
    let pb = ProgressBar::new(size)
        .with_style(ProgressStyle::with_template(
            "{msg}: {wide_bar} [{pos}/{len}] [{elapsed_precise}|{eta_precise}]"
        ).unwrap()
        ).with_message("matching words");
    if hidden { pb.set_draw_target(ProgressDrawTarget::hidden()); }
    pb
}

#[inline]
pub(crate) fn accumulate<T>(values: &[T]) -> Vec<T>
    where T: Num + NumAssignOps + Copy {
    let mut cum_values = Vec::new();
    let mut total_v = T::zero();
    for v in values {
        total_v += *v;
        cum_values.push(total_v);
    }
    cum_values
}

#[inline]
pub(crate) fn constrain<T>(value: T, min: T, max: T) -> T
    where T: Num + Copy + PartialOrd {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::accumulate;

    #[test]
    fn test_accumulate() {
        let accum = accumulate(&vec![1, 4, 4, 2]);
        assert_eq!(accum, vec![1, 5, 9, 11]);
        let accum = accumulate(&vec![0.5, -0.5, 2.0, 3.5]);
        assert_eq!(accum, vec![0.5, 0.0, 2.0, 5.5]);
        let empty: Vec<i32> = Vec::new();
        assert_eq!(accumulate(&empty), empty);
    }
}
