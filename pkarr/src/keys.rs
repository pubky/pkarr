//! Utility structs for Ed25519 keys.

use crate::{Error, Result};
use ed25519_dalek::{SecretKey, Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use std::fmt::{self, Debug, Display, Formatter};

/// Ed25519 keypair to sign dns [Packet](crate::SignedPacket)s.
pub struct Keypair(SigningKey);

impl Keypair {
    pub fn random() -> Keypair {
        let mut csprng = OsRng;
        let signing_key: SigningKey = SigningKey::generate(&mut csprng);

        Keypair(signing_key)
    }

    pub fn from_secret_key(secret_key: &SecretKey) -> Keypair {
        Keypair(SigningKey::from_bytes(secret_key))
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        self.0.sign(message)
    }

    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<()> {
        self.0.verify(message, signature)?;
        Ok(())
    }

    pub fn secret_key(&self) -> SecretKey {
        self.0.to_bytes()
    }

    pub fn public_key(&self) -> PublicKey {
        PublicKey(self.0.verifying_key())
    }

    pub fn to_z32(&self) -> String {
        self.public_key().to_string()
    }
}

/// Ed25519 public key to verify a signature over dns [Packet](crate::SignedPacket)s.
///
/// It can formatted to and parsed from a [zbase32](z32) string.
#[derive(Clone, Eq, PartialEq)]
pub struct PublicKey(VerifyingKey);

impl PublicKey {
    /// Format the public key as [zbase32](z32) string.
    pub fn to_z32(&self) -> String {
        self.to_string()
    }

    /// Verify a signature over a message.
    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<()> {
        self.0.verify(message, signature)?;
        Ok(())
    }

    /// Return a reference to the underlying [VerifyingKey](ed25519_dalek::VerifyingKey)
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.0
    }

    /// Return a the underlying [u8; 32] bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    /// Return a reference to the underlying [u8; 32] bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }
}

impl TryFrom<[u8; 32]> for PublicKey {
    type Error = ed25519_dalek::SignatureError;

    fn try_from(public: [u8; 32]) -> Result<Self, Self::Error> {
        Ok(Self(VerifyingKey::from_bytes(&public)?))
    }
}

impl TryFrom<&str> for PublicKey {
    type Error = Error;

    fn try_from(s: &str) -> Result<PublicKey> {
        let bytes =
            z32::decode(s.as_bytes()).map_err(|_| Error::Static("Invalid zbase32 encoding"))?;

        let verifying_key = VerifyingKey::try_from(bytes.as_slice())?;

        Ok(PublicKey(verifying_key))
    }
}

impl Display for PublicKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", z32::encode(self.0.as_bytes()))
    }
}

impl Display for Keypair {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.public_key())
    }
}

impl Debug for Keypair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Keypair({})", self.public_key())
    }
}

impl Debug for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PublicKey({})", self)
    }
}

#[cfg(test)]
mod tests {
    use super::Keypair;

    #[test]
    fn pkarr_key_generate() {
        let key1 = Keypair::random();
        let key2 = Keypair::from_secret_key(&key1.secret_key());

        assert_eq!(key1.public_key(), key2.public_key())
    }

    #[test]
    fn zbase32() {
        let key1 = Keypair::random();
        let _z32 = key1.public_key().to_string();

        let key2 = Keypair::from_secret_key(&key1.secret_key());

        assert_eq!(key1.public_key(), key2.public_key())
    }

    #[test]
    fn sign_verify() {
        let keypair = Keypair::random();

        let message = b"Hello, world!";
        let signature = keypair.sign(message);

        assert!(keypair.verify(message, &signature).is_ok());

        let public_key = keypair.public_key();
        assert!(public_key.verify(message, &signature).is_ok());
    }
}
