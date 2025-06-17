//! Utility structs for Ed25519 keys.

use ed25519_dalek::{
    SecretKey, Signature, SignatureError, Signer, SigningKey, Verifier, VerifyingKey,
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    hash::Hash,
    str::FromStr,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq)]
/// Ed25519 keypair to sign dns [Packet](crate::SignedPacket)s.
pub struct Keypair(pub(crate) SigningKey);

impl Keypair {
    /// Generates a new random `Keypair` using the operating system's CSPRNG.
    pub fn random() -> Keypair {
        let mut bytes = [0u8; 32];

        getrandom::getrandom(&mut bytes).expect("getrandom failed");

        let signing_key: SigningKey = SigningKey::from_bytes(&bytes);

        Keypair(signing_key)
    }

    /// Creates a `Keypair` from a given `SecretKey`.
    pub fn from_secret_key(secret_key: &SecretKey) -> Keypair {
        Keypair(SigningKey::from_bytes(secret_key))
    }

    /// Signs a message with the private key of this `Keypair`.
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.0.sign(message)
    }

    /// Verifies a message against a given signature using this `Keypair`.
    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<(), SignatureError> {
        self.0.verify(message, signature)
    }

    /// Returns the secret part of this `Keypair`.
    pub fn secret_key(&self) -> SecretKey {
        self.0.to_bytes()
    }

    /// Returns the [PublicKey] of this `Keypair`.
    pub fn public_key(&self) -> PublicKey {
        PublicKey(self.0.verifying_key())
    }

    /// Converts the public key of this `Keypair` to a z-base32 encoded string.
    pub fn to_z32(&self) -> String {
        self.public_key().to_string()
    }

    /// Converts the public key of this `Keypair` to a URI string.
    pub fn to_uri_string(&self) -> String {
        self.public_key().to_uri_string()
    }
}

/// Ed25519 public key to verify a signature over dns [Packet](crate::SignedPacket)s.
///
/// It can formatted to and parsed from a z-base32 string.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct PublicKey(pub(crate) VerifyingKey);

impl PublicKey {
    /// Format the public key as z-base32 string.
    pub fn to_z32(&self) -> String {
        self.to_string()
    }

    /// Format the public key as `pk:` URI string.
    pub fn to_uri_string(&self) -> String {
        format!("pk:{}", self)
    }

    /// Verify a signature over a message.
    pub fn verify(&self, message: &[u8], signature: &Signature) -> Result<(), SignatureError> {
        self.0.verify(message, signature)
    }

    /// Return a reference to the underlying [VerifyingKey]
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

impl AsRef<Keypair> for Keypair {
    fn as_ref(&self) -> &Keypair {
        self
    }
}

impl AsRef<PublicKey> for PublicKey {
    fn as_ref(&self) -> &PublicKey {
        self
    }
}

impl TryFrom<&[u8]> for PublicKey {
    type Error = PublicKeyError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let bytes_32: &[u8; 32] = bytes
            .try_into()
            .map_err(|_| PublicKeyError::InvalidPublicKeyLength(bytes.len()))?;

        Ok(Self(
            VerifyingKey::from_bytes(bytes_32)
                .map_err(|_| PublicKeyError::InvalidEd25519PublicKey)?,
        ))
    }
}

impl TryFrom<&[u8; 32]> for PublicKey {
    type Error = PublicKeyError;

    fn try_from(public: &[u8; 32]) -> Result<Self, Self::Error> {
        Ok(Self(
            VerifyingKey::from_bytes(public)
                .map_err(|_| PublicKeyError::InvalidEd25519PublicKey)?,
        ))
    }
}

impl From<ed25519_dalek::VerifyingKey> for PublicKey {
    fn from(public: ed25519_dalek::VerifyingKey) -> Self {
        Self(public)
    }
}

impl FromStr for PublicKey {
    type Err = PublicKeyError;

    /// Convert the TLD in a `&str` to a [PublicKey].
    ///
    /// # Examples
    ///
    /// - `o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy`
    /// - `pk:o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy`
    /// - `http://o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy`
    /// - `https://o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy`
    /// - `https://o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy/foo/bar`
    /// - `https://foo.o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy.`
    /// - `https://foo.o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy.#hash`
    /// - `https://foo@bar.o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy.?q=v`
    /// - `https://foo@bar.o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy.:8888?q=v`
    /// - `https://yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no.o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy`
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut s = s;

        if s.len() > 52 {
            // Remove scheme
            s = s.split_once(':').map(|tuple| tuple.1).unwrap_or(s);

            if s.len() > 52 {
                // Remove `//
                s = s.strip_prefix("//").unwrap_or(s);

                if s.len() > 52 {
                    // Remove username
                    s = s.split_once('@').map(|tuple| tuple.1).unwrap_or(s);

                    if s.len() > 52 {
                        // Remove port
                        s = s.split_once(':').map(|tuple| tuple.0).unwrap_or(s);

                        if s.len() > 52 {
                            // Remove trailing path
                            s = s.split_once('/').map(|tuple| tuple.0).unwrap_or(s);

                            if s.len() > 52 {
                                // Remove query
                                s = s.split_once('?').map(|tuple| tuple.0).unwrap_or(s);

                                if s.len() > 52 {
                                    // Remove hash
                                    s = s.split_once('#').map(|tuple| tuple.0).unwrap_or(s);

                                    if s.len() > 52 {
                                        if s.ends_with('.') {
                                            // Remove trailing dot
                                            s = s.trim_matches('.');
                                        }

                                        s = s.rsplit_once('.').map(|tuple| tuple.1).unwrap_or(s);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let bytes = if let Some(v) = base32::decode(base32::Alphabet::Z, s) {
            Ok(v)
        } else {
            Err(PublicKeyError::InvalidPublicKeyEncoding)
        }?;

        bytes.as_slice().try_into()
    }
}

impl TryFrom<&str> for PublicKey {
    type Error = PublicKeyError;

    fn try_from(s: &str) -> Result<PublicKey, PublicKeyError> {
        PublicKey::from_str(s)
    }
}

impl TryFrom<String> for PublicKey {
    type Error = PublicKeyError;

    fn try_from(s: String) -> Result<PublicKey, PublicKeyError> {
        s.as_str().try_into()
    }
}

impl TryFrom<&String> for PublicKey {
    type Error = PublicKeyError;

    fn try_from(s: &String) -> Result<PublicKey, PublicKeyError> {
        s.try_into()
    }
}

impl Display for PublicKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            base32::encode(base32::Alphabet::Z, self.0.as_bytes())
        )
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

impl Serialize for PublicKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes = self.to_bytes();
        bytes.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PublicKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: [u8; 32] = Deserialize::deserialize(deserializer)?;

        (&bytes).try_into().map_err(serde::de::Error::custom)
    }
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
/// Errors while trying to create a [PublicKey]
pub enum PublicKeyError {
    #[error("Invalid PublicKey length, expected 32 bytes but got: {0}")]
    /// Invalid PublicKey length.
    InvalidPublicKeyLength(usize),

    #[error("Invalid Ed25519 publickey; Cannot decompress Edwards point")]
    /// Cannot decompress Edwards point
    InvalidEd25519PublicKey,

    #[error("Invalid PublicKey encoding")]
    /// Invalid PublicKey encoding
    InvalidPublicKeyEncoding,
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn from_string() {
        let str = "yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn to_uri() {
        let bytes = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];
        let expected = "pk:yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no";

        let public_key: PublicKey = (&bytes).try_into().unwrap();

        assert_eq!(public_key.to_uri_string(), expected);
    }

    #[test]
    fn from_uri() {
        let str = "pk:yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_with_path() {
        let str = "https://yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no///foo/bar";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_with_query() {
        let str = "https://yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no?foo=bar";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_with_hash() {
        let str = "https://yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no#foo";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_with_subdomain() {
        let str = "https://foo.bar.yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no#foo";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_with_trailing_dot() {
        let str = "https://foo.yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no.";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_with_username() {
        let str = "https://foo@yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no#foo";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_with_port() {
        let str = "https://yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no:8888";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn from_uri_complex() {
        let str =
            "https://foo@bar.yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no.:8888?q=v&a=b#foo";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[test]
    fn serde() {
        let str = "yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();

        let bytes = postcard::to_allocvec(&public_key).unwrap();

        assert_eq!(bytes, expected)
    }

    #[test]
    fn from_uri_multiple_pkarr() {
        // Should only catch the TLD.

        let str =
            "https://o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy.yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no";
        let expected = [
            1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254, 14,
            207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
        ];

        let public_key: PublicKey = str.try_into().unwrap();
        assert_eq!(public_key.verifying_key().as_bytes(), &expected);
    }

    #[cfg(all(not(target_family = "wasm"), feature = "tls"))]
    #[test]
    fn pkcs8() {
        let str = "yg4gxe7z1r7mr6orids9fh95y7gxhdsxjqi6nngsxxtakqaxr5no";
        let public_key: PublicKey = str.try_into().unwrap();

        let der = public_key.to_public_key_der();

        assert_eq!(
            der.as_bytes(),
            [
                // Algorithm and other stuff.
                48, 42, 48, 5, 6, 3, 43, 101, 112, 3, 33, 0, //
                // Key
                1, 180, 103, 163, 183, 145, 58, 178, 122, 4, 168, 237, 242, 243, 251, 7, 76, 254,
                14, 207, 75, 171, 225, 8, 214, 123, 227, 133, 59, 15, 38, 197,
            ]
        )
    }

    #[cfg(all(not(target_family = "wasm"), feature = "tls"))]
    #[test]
    fn certificate() {
        use rustls::SignatureAlgorithm;

        let keypair = Keypair::from_secret_key(&[0; 32]);

        let certified_key = keypair.to_rpk_certified_key();

        assert_eq!(certified_key.key.algorithm(), SignatureAlgorithm::ED25519);

        assert_eq!(
            certified_key.end_entity_cert().unwrap().as_ref(),
            [
                48, 42, 48, 5, 6, 3, 43, 101, 112, 3, 33, 0, 59, 106, 39, 188, 206, 182, 164, 45,
                98, 163, 168, 208, 42, 111, 13, 115, 101, 50, 21, 119, 29, 226, 67, 166, 58, 192,
                72, 161, 139, 89, 218, 41,
            ]
        )
    }

    #[test]
    fn invalid_key() {
        let key = "c1bkg8tfsyy8wcedtmw4fwhdmm7bbzhgg3z58tf43m5ow8w9mbus";

        assert_eq!(
            PublicKey::try_from(key),
            Err(PublicKeyError::InvalidEd25519PublicKey)
        );
    }
}
