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

pub(crate) fn accumulate<V>(values: &[V]) -> Vec<V>
    where V: Num + NumAssignOps + Copy {
    let mut cum_values = Vec::new();
    let mut total_v = V::zero();
    for v in values {
        total_v += *v;
        cum_values.push(total_v);
    }
    cum_values
}
