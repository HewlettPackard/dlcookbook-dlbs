# __ResNet200__

Based on ResNet152. The difference is in stage 2 - we need 24 units here. ResNet152
has 8 units in 2nd stage. We need to replicate `res3b7` unit `16` times renaming it to
`res3b8`, `res3b9` ... `res3b23`.

### Architecture
```bash
  conv1
  res2a    stage 1 - units 3
  res2b
  res2c
  res3a    stage 2 - units 24
  res3b1
  ...
  res3b23
  res4a    stage 3 - units 36
  res4b1
  ...
  res4b35
  res5a    stage 4 - units 3
  res5b
  res5c
```

### Adding missing residual units
I used this [shell script](./generate_units.sh) to generate missing units
for this net as well as for ResNet269.
