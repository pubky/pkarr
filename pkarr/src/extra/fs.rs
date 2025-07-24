//! Filesystem operations involving pkarr keys.

use std::fs::{read_to_string, set_permissions, write, Permissions};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use crate::Keypair;

/// Reads the secret key from a file and derives the keypair from it.
pub fn read_keypair(secret_file_path: &Path) -> Result<Keypair, Box<dyn std::error::Error>> {
    let hex_string = read_to_string(secret_file_path)?;
    let hex_string = hex_string.trim();

    if hex_string.len() % 2 != 0 {
        return Err("Invalid hex string length".into());
    }

    let mut secret_key_bytes_vec = vec![];
    for i in (0..hex_string.len()).step_by(2) {
        let byte_str = &hex_string[i..i + 2];
        let byte = u8::from_str_radix(byte_str, 16)?;
        secret_key_bytes_vec.push(byte);
    }

    let secret_key_bytes: [u8; 32] = secret_key_bytes_vec
        .try_into()
        .map_err(|_| "Invalid secret key length")?;

    Ok(Keypair::from_secret_key(&secret_key_bytes))
}

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

#[cfg(test)]
mod tests {
    use super::{read_keypair, write_keypair};

    use std::env;
    use std::fs::{remove_file, write};

    use crate::Keypair;

    #[test]
    fn test_write_and_read_keypair() {
        let temp_file_path = env::temp_dir().join("test_keypair.tmp");

        let generated_keypair = Keypair::random();

        let write_keypair_result = write_keypair(&generated_keypair, &temp_file_path);
        assert!(write_keypair_result.is_ok());
        assert!(temp_file_path.exists());

        let read_keypair_result = read_keypair(&temp_file_path);
        assert!(read_keypair_result.is_ok());

        let read_keypair = read_keypair_result.unwrap();
        assert_eq!(generated_keypair.secret_key(), read_keypair.secret_key());

        let _ = remove_file(&temp_file_path);
    }

    #[test]
    fn test_read_keypair_invalid_hex() {
        let temp_file_path = env::temp_dir().join("test_invalid_hex.tmp");

        write(&temp_file_path, "invalidhex").unwrap();

        // Try to read file with invalid hex data
        let read_keypair_result = read_keypair(&temp_file_path);
        assert!(read_keypair_result.is_err());

        let _ = remove_file(&temp_file_path);
    }

    #[test]
    fn test_read_keypair_invalid_length() {
        let temp_file_path = env::temp_dir().join("test_invalid_length.tmp");

        write(&temp_file_path, "abcd").unwrap();

        // Try to read file with valid hex, but invalid length
        let read_keypair_result = read_keypair(&temp_file_path);
        assert!(read_keypair_result.is_err());

        let _ = remove_file(temp_file_path);
    }
}
