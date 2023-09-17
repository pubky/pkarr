use ed25519_dalek::{Signature, SignatureError, Verifier, VerifyingKey};
use reqwest;
use url::Url;
use zbase32;

struct Pkarr {
    relays: Vec<Url>,
}

impl Pkarr {
    fn new() -> Pkarr {
        Pkarr { relays: Vec::new() }
    }
    fn add_relay(&mut self, relay: &str) -> Pkarr {
        match Url::parse(relay) {
            Ok(url) => {
                self.relays.push(url);
            }
            Err(err) => {
                println!("Invalid URL: `{}`; {}", relay, err);
            }
        };

        Pkarr {
            relays: self.relays.clone(),
        }
    }

    fn resolve(&self, id: &str) {
        let resp = reqwest::blocking::get(self.relays[0].join(id).unwrap());

        match resp {
            Ok(resp) => {
                if resp.status() != 200 {
                    println!("Could not get a valid response from the relay");
                    return;
                }

                match resp.bytes() {
                    Ok(bytes) => {
                        let mut signature = [0; 64];
                        signature.copy_from_slice(&bytes.slice(0..64));

                        let decoded = match zbase32::decode_full_bytes(id.as_bytes()) {
                            Ok(pk) => pk,
                            Err(_) => {
                                println!("Invalid public_key");
                                return;
                            }
                        };

                        let mut public_key = [0_u8; 32];
                        public_key.copy_from_slice(&decoded);

                        let message = bytes.slice(64..);
                        let verified = verify(&signature, &message, &public_key);

                        println!("{:?}", verified);

                        assert!(verified.is_ok());
                    }
                    Err(_) => println!("Could not read text from response"),
                }
            }
            Err(_) => println!("Could not make the damn request!"),
        }
    }
}

fn verify(
    signature: &[u8; 64],
    message: &[u8],
    public_key: &[u8; 32],
) -> Result<(), SignatureError> {
    let verifying_key = VerifyingKey::from_bytes(&public_key).unwrap();

    let sig = Signature::from_bytes(&signature);

    verifying_key.verify(message, &sig)
}

fn main() {
    let pkarr = Pkarr::new().add_relay("https://relay.pkarr.org");

    let url = "pk:o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy";

    let id = match url.split(":").nth(1) {
        Some(id) => id,
        None => {
            println!("Invalid URL format");
            return;
        }
    };

    pkarr.resolve(id)
}
