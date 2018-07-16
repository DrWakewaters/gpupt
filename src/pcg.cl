// A random number generator. See https://en.wikipedia.org/wiki/Permuted_congruential_generator.
unsigned int next_uint(unsigned long *restrict state) {
    unsigned long x = *state;
    unsigned int count = (unsigned int)(x >> 59);
    *state = x*multiplier + increment;
    x ^= x >> 18;
    return (((unsigned int)(x >> 27)) >> count) | (((unsigned int)(x >> 27)) << (-count & 31));
}

// A random number in [inclusive_minimum, exclusive_maximum).
unsigned int next_uint_in_range(unsigned int inclusive_minimum, unsigned int exclusive_maximum, unsigned long *restrict state) {
    unsigned int a = exclusive_maximum-inclusive_minimum;
    unsigned int b = UINT_MAX-(UINT_MAX%a);
    while(true) {
        unsigned int random = next_uint(state);
        if(random < b) {
            return (random%a)+inclusive_minimum;
        }
    }
}

float next_float(unsigned long *restrict state) {
    unsigned int random_uint = next_uint(state);
    return ((float)random_uint)/((float)UINT_MAX);
}

void pcg_init(unsigned long seed, unsigned long *restrict state) {
    *state = 2*seed + 1;
    (void)next_uint(state);
}
