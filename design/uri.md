# URI

While not necessary, it is advised to use the following URI to represent Public keys with Pkarr records:

```
pk:<52 character zbase-32 encoded public-key>
```

Implementations should be able to parse both `pk:<zbase32 encoded key>` and `<zbase-32 encoded key>`.
