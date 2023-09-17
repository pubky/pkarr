use reqwest;

fn main() {
    let resp = reqwest::blocking::get(
        "https://relay.pkarr.org/o4dksfbqk85ogzdb5osziw6befigbuxmuxkuxq8434q89uj56uyy",
    );

    match resp {
        Ok(resp) => match resp.bytes() {
            Ok(bytes) => {
                let signature = bytes.slice(0..64);
                let message = bytes.slice(64..);
                println!("{:?} {:?}", signature.len(), message.len())
            }
            Err(_) => println!("Could not read text from response"),
        },
        Err(_) => println!("Could not make the damn request!"),
    }

    println!("Hello, world!");
}
