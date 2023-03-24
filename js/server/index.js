import http from 'http';
import url from 'url';
import DHT from 'bittorrent-dht';
import ed from 'bittorrent-dht-sodium'

const dht = new DHT({ verify: ed.verify  });

const server = http.createServer(async (req, res) => {
  boilerplate(req, res);

  const parsedUrl = url.parse(req.url, true);
  const path = parsedUrl.pathname.split('/');
  const hash = path[2];

  if (!path[1] === 'pkarr' || !hash) {
    res.statusCode = 404;
    res.end('Not found');
    return
  }

  // Handle GET /pkarr/:hash
  if (req.method === 'GET') {
    dht.get(hash, (err, response) => {
      if (err) {
        res.statusCode = 500;
        res.end('Failed to fetch value from DHT');
        return;
      }

      if (!response) {
        res.statusCode = 404;
        res.end('Value not found');
        return;
      }

      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({
		    k: response.k.toString('hex'),
		    s: response.seq || 0,
		    v: response.v.toString('hex'),
	    }));
    });
  // Handle PUT /pkarr/:hash
  } else if (req.method === 'PUT') {
    // let body = '';
    //
    // req.on('data', chunk => {
    //   body += chunk;
    // });
    //
    // req.on('end', async () => {
    //   const payload = JSON.parse(body);
    //
    //   dht.put({ v: payload.value }, (err, hash) => {
    //     if (err) {
    //       res.statusCode = 500;
    //       res.end('Failed to put value into DHT');
    //       return;
    //     }
    //
    //     res.setHeader('Content-Type', 'application/json');
    //     res.end(JSON.stringify({ hash: hash.toString('hex') }));
    //   });
    // });
  } else {
    res.statusCode = 405;
    res.end('Method not allowed');
  }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server is listening on port ${PORT}`);
});

function boilerplate (req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Preflight request. Reply successfully:
  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }
}
