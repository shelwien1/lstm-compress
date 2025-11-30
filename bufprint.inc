
// Double-buffered output (128k total = 64k * 2)
enum { output_bufsize = 64*1024 };
static char output_buffer[output_bufsize * 2];
static uint output_pos = 0;
static FILE* output_file = 0;

// ----------------------------- minimal type traits (no <type_traits>) --------------
template<typename T> struct is_floating_point_ { static constexpr bool value = false; };
template<> struct is_floating_point_<float>  { static constexpr bool value = true; };
template<> struct is_floating_point_<double> { static constexpr bool value = true; };
template<> struct is_floating_point_<long double> { static constexpr bool value = true; };

template<typename T> struct is_signed_integral_ { static constexpr bool value = false; };
template<> struct is_signed_integral_<signed char> { static constexpr bool value = true; };
template<> struct is_signed_integral_<short>       { static constexpr bool value = true; };
template<> struct is_signed_integral_<int>         { static constexpr bool value = true; };
template<> struct is_signed_integral_<long>        { static constexpr bool value = true; };
template<> struct is_signed_integral_<long long>   { static constexpr bool value = true; };

template<typename T> struct is_unsigned_integral_ { static constexpr bool value = false; };
template<> struct is_unsigned_integral_<unsigned char> { static constexpr bool value = true; };
template<> struct is_unsigned_integral_<unsigned short> { static constexpr bool value = true; };
template<> struct is_unsigned_integral_<unsigned int> { static constexpr bool value = true; };
template<> struct is_unsigned_integral_<unsigned long> { static constexpr bool value = true; };
template<> struct is_unsigned_integral_<unsigned long long> { static constexpr bool value = true; };

template<typename T> struct is_cstring_ { static constexpr bool value = false; };
template<> struct is_cstring_<char*> { static constexpr bool value = true; };
template<> struct is_cstring_<const char*> { static constexpr bool value = true; };


// Flush output buffer
void flush_output() {
  if (output_file && output_pos > 0) {
    fwrite(output_buffer, 1, output_pos, output_file);
    output_pos = 0;
  }
}

// Custom integer to string conversion
NOINLINE
static int int_to_str(char* buf, qword val, int width, bool zero_pad) {
  char tmp[32];
  int len = 0;

  if (val == 0) {
    tmp[len++] = '0';
  } else {
    while (val > 0) {
      tmp[len++] = '0' + (val % 10);
      val /= 10;
    }
  }

  int pad_len = len;
  while (pad_len < width) {
    tmp[pad_len++] = zero_pad ? '0' : ' ';
  }

  for (int i = 0; i < pad_len; i++) {
    buf[i] = tmp[pad_len - 1 - i];
  }

  return pad_len;
}

// Custom hexadecimal to string conversion (uppercase)
NOINLINE
static int hex_to_str(char* buf, qword val, int width, bool zero_pad) {
  char tmp[32];
  int len = 0;

  if (val == 0) {
    tmp[len++] = '0';
  } else {
    while (val > 0) {
      int digit = val % 16;
      tmp[len++] = (digit < 10) ? ('0' + digit) : ('A' + digit - 10);
      val /= 16;
    }
  }

  int pad_len = len;
  while (pad_len < width) {
    tmp[pad_len++] = zero_pad ? '0' : ' ';
  }

  for (int i = 0; i < pad_len; i++) {
    buf[i] = tmp[pad_len - 1 - i];
  }

  return pad_len;
}

// Custom double to string conversion
NOINLINE
static int double_to_str(char* buf, double val, int precision) {
  int pos = 0;

  if (val < 0) {
    buf[pos++] = '-';
    val = -val;
  }

  qword int_part = (qword)val;
  pos += int_to_str(buf + pos, int_part, 0, false);

  if (precision > 0) {
    buf[pos++] = '.';
    double frac = val - int_part;
    for (int i = 0; i < precision; i++) {
      frac *= 10;
      int digit = (int)frac;
      buf[pos++] = '0' + digit;
      frac -= digit;
    }
  }

  return pos;
}

//============================================================================
// C++17 compile-time format string via auto& NTTP
//============================================================================

// Helper to get string length excluding null terminator
template<size_t N>
constexpr size_t str_len(const char (&)[N]) { return N - 1; }

//----------------------------------------------------------------------------
// Write accumulated literal characters
//----------------------------------------------------------------------------

template<char... Cs>
inline char* write_chars(char* out) {
  if constexpr (sizeof...(Cs) == 0) {
    return out;
  } else if constexpr (sizeof...(Cs) <= 8) {
    constexpr char arr[] = { Cs... };
    for (size_t i = 0; i < sizeof...(Cs); i++) out[i] = arr[i];
    return out + sizeof...(Cs);
  } else {
    static constexpr char arr[] = { Cs... };
    memcpy(out, arr, sizeof...(Cs));
    return out + sizeof...(Cs);
  }
}

//----------------------------------------------------------------------------
// Compile-time format spec parser
//----------------------------------------------------------------------------

template<const auto& Fmt, size_t I>
struct SpecParser {
  static constexpr size_t len = str_len(Fmt);
  static constexpr bool zero_pad = (I < len && Fmt[I] == '0');
  static constexpr size_t after_zero = zero_pad ? I + 1 : I;

  static constexpr int parse_width_impl(size_t pos, int acc) {
    if (pos >= len) return acc;
    char c = Fmt[pos];
    if (c >= '0' && c <= '9') return parse_width_impl(pos + 1, acc * 10 + (c - '0'));
    return acc;
  }
  static constexpr size_t skip_width_impl(size_t pos) {
    if (pos >= len) return pos;
    char c = Fmt[pos];
    if (c >= '0' && c <= '9') return skip_width_impl(pos + 1);
    return pos;
  }
  static constexpr int width = parse_width_impl(after_zero, 0);
  static constexpr size_t after_width = skip_width_impl(after_zero);

  static constexpr bool has_precision = (after_width < len && Fmt[after_width] == '.');
  static constexpr size_t prec_start = has_precision ? after_width + 1 : after_width;
  static constexpr int precision = has_precision ? parse_width_impl(prec_start, 0) : -1;
  static constexpr size_t after_prec = has_precision ? skip_width_impl(prec_start) : after_width;

  static constexpr size_t skip_length_impl(size_t pos) {
    if (pos >= len) return pos;
    char c = Fmt[pos];
    if (c == 'l' || c == 'h') return skip_length_impl(pos + 1);
    return pos;
  }
  static constexpr size_t after_length = skip_length_impl(after_prec);

  static constexpr char type = (after_length < len) ? Fmt[after_length] : '\0';
  static constexpr size_t end_pos = (after_length < len) ? after_length + 1 : after_length;
};

//----------------------------------------------------------------------------
// Formatter struct - handles mutual recursion via nested templates
//----------------------------------------------------------------------------

template<const auto& Fmt>
struct Formatter {
  static constexpr size_t len = str_len(Fmt);
  
  // Accumulate trailing literal chars (no more args)
  template<size_t I, char... Accum>
  static inline char* parse_tail(char* out) {
    if constexpr (I >= len) {
      return write_chars<Accum...>(out);
    } else {
      return parse_tail<I + 1, Accum..., Fmt[I]>(out);
    }
  }

  // Format one value based on spec at position I
  template<size_t I, typename T>
  static inline char* format_one(char* out, T val) {
    using Spec = SpecParser<Fmt, I>;
    
    constexpr bool is_signed_int = is_signed_integral_<T>::value;
    constexpr bool is_unsigned_int = is_unsigned_integral_<T>::value;
    constexpr bool is_float = is_floating_point_<T>::value;

    if constexpr (is_float) {
      constexpr int prec = (Spec::precision >= 0) ? Spec::precision : 6;
      out += double_to_str(out, (double)val, prec);
    } else if constexpr (is_signed_int) {
      sqword sval = (sqword)val;
      qword uval;
      if (sval < 0) {
        *out++ = '-';
        uval = (qword)(-sval);
      } else {
        uval = (qword)sval;
      }
      if constexpr (Spec::type == 'X' || Spec::type == 'x') {
        out += hex_to_str(out, uval, Spec::width, Spec::zero_pad);
      } else {
        out += int_to_str(out, uval, Spec::width, Spec::zero_pad);
      }
    } else if constexpr (is_unsigned_int) {
      if constexpr (Spec::type == 'X' || Spec::type == 'x') {
        out += hex_to_str(out, (qword)val, Spec::width, Spec::zero_pad);
      } else {
        out += int_to_str(out, (qword)val, Spec::width, Spec::zero_pad);
      }
    }
    return out;
  }

  // Main parser - accumulates literal chars, dispatches on %
  template<size_t I, char... Accum, typename T, typename... Rest>
  static inline char* parse(char* out, T val, Rest... rest) {
    if constexpr (I >= len) {
      return write_chars<Accum...>(out);
    } else if constexpr (Fmt[I] == '%') {
      // Write accumulated literals
      out = write_chars<Accum...>(out);
      // Format this value
      using Spec = SpecParser<Fmt, I + 1>;
      out = format_one<I + 1>(out, val);
      // Continue with remaining args
      if constexpr (sizeof...(Rest) > 0) {
        return parse<Spec::end_pos>(out, rest...);
      } else {
        return parse_tail<Spec::end_pos>(out);
      }
    } else {
      // Accumulate literal char
      return parse<I + 1, Accum..., Fmt[I]>(out, val, rest...);
    }
  }

  // Entry with no args - just literals
  template<size_t I, char... Accum>
  static inline char* parse(char* out) {
    return parse_tail<I, Accum...>(out);
  }
};

//----------------------------------------------------------------------------
// Entry points
//----------------------------------------------------------------------------

template<const auto& Fmt, typename... Args>
inline int buffered_sprintf(char* out, Args... args) {
  char* end;
  if constexpr (sizeof...(Args) == 0) {
    end = Formatter<Fmt>::template parse_tail<0>(out);
  } else {
    end = Formatter<Fmt>::template parse<0>(out, args...);
  }
  return (int)(end - out);
}

template<const auto& Fmt, typename... Args>
void buffered_printf(Args... args) {
  char temp[256];
  int len = buffered_sprintf<Fmt>(temp, args...);

  if (output_pos >= output_bufsize && output_pos < output_bufsize * 2) {
    if (output_file) {
      fwrite(output_buffer, 1, output_bufsize, output_file);
    }
    uint overflow = output_pos - output_bufsize;
    if (overflow > 0) {
      memcpy(output_buffer, output_buffer + output_bufsize, overflow);
    }
    output_pos = overflow;
  } else if (output_pos >= output_bufsize * 2) {
    flush_output();
  }

  memcpy(output_buffer + output_pos, temp, len);
  output_pos += len;
}

// Convenience macro for inline format strings
#define BPRINTF(fmt, ...) \
  do { static constexpr char _fmt_[] = fmt; buffered_printf<_fmt_>(__VA_ARGS__); } while(0)

#if 0
//============================================================================

//static constexpr char fmt1[] = "%06i_%06i - %016X - %.3lf\n";

int main( int argc, char** argv ) {

  struct Result {
    int i, j;
    qword csize;
    double ctime;
  };

  Result res = { 123, 4567, 89101112+(1LL<<62), 123.456789 };

  // Option 1: explicit static constexpr format string
  //buffered_printf<fmt1>( res.i, res.j, res.csize, res.ctime );

  // Option 2: convenience macro (inline format string)
  //BPRINTF( "%06i_%06i - %016X - %.3lf\n", res.i, res.j, res.csize, res.ctime );
  BPRINTF("Hello, world!!! %06i_%06i - %016X - %.3lf\n", res.i, res.j, res.csize, res.ctime );

  fwrite( output_buffer, 1, output_pos, stdout );

  return 0;
}
#endif
