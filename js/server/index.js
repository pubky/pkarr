import DHT from 'bittorrent-dht'
import sodium from 'sodium-native'

const dht = new DHT({
  verify: sodium.crypto_sign_verify_detached
})

    console.log(dht.table)
dht.get(
  Buffer.from('3742a9a6bed50ea6899269623af5b5431d22f905', 'hex'), 
  function (err, hash) {
    console.log(err, hash)
  }
)

// Optionally pass an exisiting secret key and rederive the public key
function generateKeyPair (secretKey) {
  const publicKey = Buffer.alloc(sodium.crypto_sign_PUBLICKEYBYTES)
  if (secretKey == null) {
    secretKey = sodium.sodium_malloc(sodium.crypto_sign_SECRETKEYBYTES)
    sodium.crypto_sign_keypair(publicKey, secretKey)
  } else {
    sodium.crypto_sign_ed25519_sk_to_pk(publicKey, secretKey)
  }

  return { publicKey, secretKey }
}

function sign (msg) {
  const sig = Buffer.alloc(sodium.crypto_sign_BYTES)
  sodium.crypto_sign_detached(sig, msg, keyPair.secretKey)
  return sig
}
