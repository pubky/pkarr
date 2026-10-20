use std::{fmt, str::FromStr};

use crate::{Error, PublicKey};

/// An exact service name in a Pubky key's namespace.
///
/// ASCII case and one trailing root dot are normalized. URLs, wildcards,
/// Unicode, empty labels, and noncanonical key encodings are rejected.
/// Underscores are allowed for names such as `_pubky.<key>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "service-protocol", derive(serde::Deserialize))]
#[cfg_attr(feature = "service-protocol", serde(try_from = "String"))]
pub struct ServiceName {
    name: String,
    key: PublicKey,
}

#[cfg(feature = "service-protocol")]
impl serde::Serialize for ServiceName {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "service-protocol")]
impl From<ServiceName> for String {
    fn from(service: ServiceName) -> Self {
        service.name
    }
}

#[cfg(feature = "service-protocol")]
impl TryFrom<String> for ServiceName {
    type Error = Error;

    fn try_from(name: String) -> Result<Self, Error> {
        name.parse()
    }
}

impl ServiceName {
    /// The normalized DNS name, without a trailing root dot.
    pub fn as_str(&self) -> &str {
        &self.name
    }

    /// The key whose packet may authorize records under this name.
    pub fn public_key(&self) -> &PublicKey {
        &self.key
    }
}

impl FromStr for ServiceName {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self, Error> {
        let name = input
            .strip_suffix('.')
            .unwrap_or(input)
            .to_ascii_lowercase();
        let valid = !name.is_empty()
            && name.len() <= 253
            && name.is_ascii()
            && name.split('.').all(|label| {
                !label.is_empty()
                    && label.len() <= 63
                    && !label.starts_with('-')
                    && !label.ends_with('-')
                    && label
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            });
        if !valid {
            return Err(Error::InvalidServiceName(input.to_owned()));
        }
        let key_label = name.rsplit('.').next().expect("nonempty validated name");
        let key: PublicKey = key_label
            .parse()
            .map_err(|_| Error::InvalidServiceName(input.to_owned()))?;
        if key.to_z32() != key_label {
            return Err(Error::InvalidServiceName(input.to_owned()));
        }
        Ok(Self { name, key })
    }
}

impl fmt::Display for ServiceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

#[cfg(test)]
mod tests {
    use pkarr::Keypair;

    use super::*;

    #[test]
    fn validation_preserves_scopes_and_rejects_urls_and_wildcards() {
        let key = Keypair::from_secret_key(&[1; 32]).to_z32();
        let upper = format!("APP.{}.", key.to_ascii_uppercase());
        let parsed: ServiceName = upper.parse().unwrap();
        assert_eq!(parsed.as_str(), format!("app.{key}"));
        for prefix in ["https://", "*.", ".", "-app.", "app..", "é."] {
            assert!(format!("{prefix}{key}").parse::<ServiceName>().is_err());
        }
    }
}
