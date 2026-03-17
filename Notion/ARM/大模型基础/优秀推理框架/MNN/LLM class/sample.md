---

---
```javascript
int Llm::sample(VARP logits, int offset, int size) {
    auto logitsShape = logits->getInfo()->dim;
    if (offset && size) {
        MNN_ASSERT(logits->getInfo()->size >= offset + size);
        logits = _Const(logits->readMap<float>() + offset, {size}, NHWC, halide_type_of<float>());
    }
    auto token_id = mSampler->sample(logits);
    return token_id;
}
```
