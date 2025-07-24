//! Filesystem operations involving pkarr keys.

use std::fs::{read_to_string, set_permissions, write, Permissions};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use crate::Keypair;

/// Writes the secret key of the keypair to file, as a hex encoded string.
/// If the file already exists, it will be overwritten.
pub fn write_keypair(keypair: &Keypair, secret_file_path: &Path) -> Result<(), std::io::Error> {
    let secret = keypair.secret_key();
    let hex_string: String = secret.iter().map(|b| format!("{:02x}", b)).collect();
    write(secret_file_path, hex_string)?;
    #[cfg(unix)]
    {
        set_permissions(&secret_file_path, Permissions::from_mode(0o600))?;
    }
    Ok(())
}
