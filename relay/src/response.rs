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
        if let Some(date) = self.signed_packet.timestamp().format_http_date() {
            headers.insert(header::LAST_MODIFIED, date.try_into()?);
        }
        headers.insert(
            "memento-datetime",
            self.signed_packet
                .last_seen()
                .format_http_date_saturating()
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
    use std::time::SystemTime;

    use super::*;
    use pkarr::Keypair;

    fn signed_packet(timestamp: Timestamp) -> SignedPacket {
        SignedPacket::builder()
            .timestamp(timestamp)
            .sign(&Keypair::random())
            .unwrap()
    }

    #[tokio::test]
    async fn out_of_range_dates_do_not_prevent_serving_the_packet() {
        let mut packet = signed_packet(Timestamp::from(u64::MAX));
        packet.set_last_seen(&Timestamp::from(u64::MAX));
        let payload = packet.to_relay_payload();

        let condition = "Fri, 31 Dec 9999 23:59:59 GMT".parse().unwrap();
        let response = SignedPacketResponse::new(packet, 30, Some(condition)).into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(!response.headers().contains_key(header::LAST_MODIFIED));
        assert_eq!(
            response.headers()["memento-datetime"],
            "Fri, 31 Dec 9999 23:59:59 GMT"
        );
        assert_eq!(
            response.headers()[header::CACHE_CONTROL],
            "public, max-age=30"
        );
        let body = axum::body::to_bytes(response.into_body(), SignedPacket::MAX_BYTES as usize)
            .await
            .unwrap();
        assert_eq!(body, payload);
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
            Some(HttpDate::from(SystemTime::from(condition))),
            &signed_packet(packet_timestamp)
        ));
    }

    #[test]
    fn not_modified_still_matches_exact_http_second_timestamp() {
        let packet_timestamp = Timestamp::parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT").unwrap();

        assert!(is_not_modified(
            Some(HttpDate::from(SystemTime::from(packet_timestamp))),
            &signed_packet(packet_timestamp)
        ));
    }
}
