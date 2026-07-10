use axum::extract::FromRequestParts;
use http::{header, request::Parts};
use httpdate::HttpDate;
use pkarr::PublicKey;
use serde::{de::Error, Deserialize, Deserializer};
use std::str::FromStr;

use crate::error::Error as RelayError;

pub(crate) struct PublicKeyParam(pub PublicKey);

impl<'de> Deserialize<'de> for PublicKeyParam {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let public_key = String::deserialize(deserializer)?;

        PublicKey::try_from(public_key.as_str())
            .map(Self)
            .map_err(D::Error::custom)
    }
}

pub(crate) struct IfModifiedSince(pub Option<HttpDate>);

impl<S> FromRequestParts<S> for IfModifiedSince
where
    S: Send + Sync,
{
    type Rejection = RelayError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        Ok(Self(
            parts
                .headers
                .get(header::IF_MODIFIED_SINCE)
                .and_then(|h| h.to_str().ok())
                .and_then(|s| HttpDate::from_str(s).ok()),
        ))
    }
}
