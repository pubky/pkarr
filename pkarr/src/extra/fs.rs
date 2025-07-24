//! Filesystem operations involving pkarr keys.

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use crate::Keypair;

/// Writes the secret key of the keypair to file, as a hex encoded string.
/// If the file already exists, it will be overwritten.
pub fn write_keypair(keypair: &Keypair, secret_file_path: &Path) -> Result<(), std::io::Error> {
    let secret = keypair.secret_key();
    let hex_string = const_hex::encode(secret);
    std::fs::write(secret_file_path.clone(), hex_string)?;
    #[cfg(unix)]
    {
        std::fs::set_permissions(&secret_file_path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}
