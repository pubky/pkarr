use axum::{http::HeaderMap, response::IntoResponse};
use http::{
    header::{self, InvalidHeaderValue},
    StatusCode,
};
use httpdate::HttpDate;
use pkarr::{SignedPacket, StoredNodeCount, Timestamp, PKARR_DHT_STORED_NODES};

pub(crate) struct PutResponse {
    stored_on: StoredNodeCount,
}

impl PutResponse {
    pub(crate) fn new(stored_on: StoredNodeCount) -> Self {
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
        let condition = Timestamp::from(condition_http_date);

        signed_packet.timestamp() <= condition
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pkarr::Keypair;

    fn signed_packet(timestamp: Timestamp) -> SignedPacket {
        SignedPacket::builder()
            .timestamp(timestamp)
            .sign(&Keypair::random())
            .unwrap()
    }

    #[test]
    fn not_modified_uses_full_packet_timestamp_precision() {
        let http_second = Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();
        let packet_timestamp = http_second + 500_000;
        let condition = packet_timestamp - 1;
        assert_eq!(
            condition.format_http_date(),
            packet_timestamp.format_http_date()
        );

        assert!(!is_not_modified(
            Some(HttpDate::from(condition)),
            &signed_packet(packet_timestamp)
        ));
    }

    #[test]
    fn not_modified_still_matches_exact_http_second_timestamp() {
        let packet_timestamp = Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();

        assert!(is_not_modified(
            Some(HttpDate::from(packet_timestamp)),
            &signed_packet(packet_timestamp)
        ));
    }
}
