use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

pub fn get_progress_bar(size: u64, hidden: bool) -> ProgressBar {
    let pb = ProgressBar::new(size)
        .with_style(ProgressStyle::with_template(
            "{msg}: {wide_bar} [{pos}/{len}] [{elapsed_precise}|{eta_precise}]"
        ).unwrap()
        ).with_message("matching words");
    if hidden { pb.set_draw_target(ProgressDrawTarget::hidden()); }
    pb
}
