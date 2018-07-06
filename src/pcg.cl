// A random number generator. See https://en.wikipedia.org/wiki/Permuted_congruential_generator
unsigned int next_uint(unsigned long *restrict state) {
    unsigned long x = *state;
    unsigned int count = (unsigned int)(x >> 59);
    *state = x*multiplier + increment;
    x ^= x >> 18;
    return (((unsigned int)(x >> 27)) >> count) | (((unsigned int)(x >> 27)) << (-count & 31));
}

float next_float(unsigned long *restrict state) {
    unsigned int random_uint = next_uint(state);
    return ((float)random_uint)/((float)UINT_MAX);
}

void pcg_init(unsigned long seed, unsigned long *restrict state) {
    *state = 2*seed + 1;
    (void)next_uint(state);
}
