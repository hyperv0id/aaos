use std::path::{Path, PathBuf};

/// Resolve a tool path argument against the session cwd: absolute paths pass
/// through unchanged, relative paths are joined onto `cwd`.
pub fn resolve_to_cwd(path: &str, cwd: &Path) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn relative_joins_cwd() {
        let p = resolve_to_cwd("src/lib.rs", Path::new("/tmp/proj"));
        assert_eq!(p, Path::new("/tmp/proj/src/lib.rs"));
    }

    #[test]
    fn absolute_is_unchanged() {
        let p = resolve_to_cwd("/etc/hosts", Path::new("/tmp/proj"));
        assert_eq!(p, Path::new("/etc/hosts"));
    }
}
