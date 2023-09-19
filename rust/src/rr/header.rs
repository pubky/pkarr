// Resource Record name can be encoded in three ways:
// 1. Origin: The name equals the origin of the zone, in this case the public key of Pkarr
//    (equivilant to '@' in DNS zone file), in which case we use zero extra bytes to represent the name.
// 2. Compressed: The name is the same as in a previous Resource Record in the same file, in
//    which case we only use a single extra byte (the index of the first resource record with a matching name) to represent the name.
// 3. LengthValue: The name is encoded as a utf-8 string and an extra byte is used to represent the
//    length of the string.
#[derive(Debug)]
pub enum RRHeaderNameEncoding {
    Origin,
    Compressed,
    LengthValue,
}

pub fn encode_rr_header(
    out: &mut Vec<u8>,
    name_encoding: RRHeaderNameEncoding,
    rtype: u16,
    ttl: u32,
) -> u8 {
    let mut header_byte: u8 = 0;

    // Set the second and third bits.
    // The second bit is a flag for a Compressed encoding.
    // The third bit is a flag for a LengthValue encoding.
    match name_encoding {
        RRHeaderNameEncoding::Origin => header_byte |= 0b0000_0000,
        RRHeaderNameEncoding::LengthValue => header_byte |= 0b0010_0000,
        RRHeaderNameEncoding::Compressed => header_byte |= 0b0100_0000,
    }

    let rtype_len = if rtype > 0xff {
        2
    } else if rtype > 0 {
        1
    } else {
        0
    };

    header_byte |= rtype_len;

    let ttl_len = if ttl > 0xffffff {
        4
    } else if ttl > 0xffff {
        3
    } else if ttl > 0xff {
        2
    } else if ttl > 0 {
        1
    } else {
        0
    };

    header_byte |= ttl_len << 2;

    out.push(header_byte);

    header_byte
}

pub fn name_encoding(header: u8) -> RRHeaderNameEncoding {
    let encoding: u8 = (header & 0b0110_000) >> 5;

    match encoding {
        0 => return RRHeaderNameEncoding::Origin,
        1 => return RRHeaderNameEncoding::LengthValue,
        _ => return RRHeaderNameEncoding::Compressed,
    }
}

pub fn rtype_len(header: u8) -> u8 {
    header & 0b0000_0011
}

pub fn ttl_len(header: u8) -> u8 {
    (header & 0b0001_1100) >> 2
}

#[cfg(test)]
mod tests {
    use crate::rr::header::*;

    #[test]
    fn resource_records_header_encode_and_parse() {
        let mut out: Vec<u8> = vec![];

        let header = encode_rr_header(&mut out, RRHeaderNameEncoding::Origin, 1, 3600);

        println!("{:?}", header);
        println!(
            "{:?}, {:?},  {:?}",
            name_encoding(header),
            rtype_len(header),
            ttl_len(header)
        )
    }
}
