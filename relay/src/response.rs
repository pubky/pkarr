use axum::{http::HeaderMap, response::IntoResponse};
use http::{
    header::{self, InvalidHeaderValue},
    StatusCode,
};
use httpdate::HttpDate;
use pkarr::{SignedPacket, PKARR_DHT_STORED_NODES};

pub(crate) struct PutResponse {
    stored_on: u32,
}

impl PutResponse {
    pub(crate) fn new(stored_on: u32) -> Self {
        Self { stored_on }
    }
}

impl IntoResponse for PutResponse {
    fn into_response(self) -> axum::response::Response {
        (
            StatusCode::NO_CONTENT,
            [(PKARR_DHT_STORED_NODES, self.stored_on.to_string())],
        )
            .into_response()
    }
}

pub(crate) struct SignedPacketResponse {
    signed_packet: SignedPacket,
    ttl: u32,
    not_modified: bool,
}

impl SignedPacketResponse {
    pub fn new(signed_packet: SignedPacket, ttl: u32, if_modified_since: Option<HttpDate>) -> Self {
        let not_modified = is_not_modified(if_modified_since, &signed_packet);
        Self {
            signed_packet,
            ttl,
            not_modified,
        }
    }

    fn headers(&self) -> Result<HeaderMap, InvalidHeaderValue> {
        let mut headers = HeaderMap::with_capacity(4);

        headers.insert(
            header::CONTENT_TYPE,
            "application/pkarr.org/relays#payload".try_into()?,
        );
        headers.insert(
            header::CACHE_CONTROL,
            format!("public, max-age={}", self.ttl).try_into()?,
        );
        headers.insert(
            header::LAST_MODIFIED,
            self.signed_packet
                .timestamp()
                .format_http_date()
                .try_into()?,
        );
        headers.insert(
            "memento-datetime",
            self.signed_packet
                .last_seen()
                .format_http_date()
                .try_into()?,
        );

        Ok(headers)
    }
}

impl IntoResponse for SignedPacketResponse {
    fn into_response(self) -> axum::response::Response {
        let mut response = match self.headers() {
            Ok(headers) => headers.into_response(),
            Err(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to build response headers: {error}"),
                )
                    .into_response()
            }
        };

        if self.not_modified {
            *response.status_mut() = StatusCode::NOT_MODIFIED;
        } else {
            *response.body_mut() = self.signed_packet.to_relay_payload().into();
        }

        response
    }
}

fn is_not_modified(if_modified_since: Option<HttpDate>, signed_packet: &SignedPacket) -> bool {
    if_modified_since.is_some_and(|condition_http_date| {
        let entry_http_date: HttpDate = signed_packet.timestamp().into();

        condition_http_date >= entry_http_date
    })
}
