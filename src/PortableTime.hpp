#pragma once

// @adai-status: beta        (TD-160 — new, no test coverage yet beyond the callers it replaces)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-12

/**
 * @file PortableTime.hpp
 * @brief Portable replacements for POSIX-only UTC time functions used across this codebase
 * (gmtime_r, timegm, strptime) that have no equivalent — or a differently-shaped one — on
 * MinGW/Windows (TD-160). Centralized here instead of repeating #ifdef _WIN32 at each of the
 * ~8 call sites across src/DaemonConfigStore.cpp, DatasetRegistry.cpp, SQLiteMetricsDatabase.cpp,
 * TrainingMetricsAPI.cpp, RegistryServer.cpp, FtpDataServer.hpp, ModelNameService.cpp, and
 * PostgresMetricsDatabase.cpp, all of which parse/format UTC timestamps the same way.
 */

#include <cstdio>
#include <ctime>

namespace adai {

/**
 * Portable equivalent of POSIX gmtime_r(time, result) — converts *time to broken-down UTC time
 * into *result, returning result on success or nullptr on failure.
 *
 * Windows' CRT has no gmtime_r, but does have gmtime_s — same behavior, reversed argument order
 * (result first) and an errno_t return instead of a struct tm*, wrapped here to match gmtime_r's
 * own signature/return convention so callers don't need their own #ifdef.
 */
inline std::tm* gmtime_utc(const std::time_t* time, std::tm* result) {
#ifdef _WIN32
    return (gmtime_s(result, time) == 0) ? result : nullptr;
#else
    return gmtime_r(time, result);
#endif
}

/**
 * Portable equivalent of POSIX timegm(tm) — interprets *tm as UTC (not local time, unlike
 * mktime()) and returns the corresponding time_t.
 *
 * Windows' CRT has no timegm, but does have the functionally-identical _mkgmtime under a
 * different name.
 */
inline std::time_t timegm_utc(std::tm* tm) {
#ifdef _WIN32
    return _mkgmtime(tm);
#else
    return timegm(tm);
#endif
}

/**
 * Portable equivalent of POSIX strptime(s, fmt, tm) for this codebase's only two formats in
 * actual use: "YYYY-MM-DDTHH:MM:SS" and "YYYY-MM-DD HH:MM:SS" — pass `separator` as 'T' or ' '
 * respectively. Returns a pointer to the first unconsumed character on success (matching
 * strptime's own return convention), or nullptr if `s` doesn't match.
 *
 * strptime() has no Windows/MinGW equivalent at all (unlike gmtime_r/timegm above, which have a
 * same-behavior, differently-named CRT counterpart) — this is a from-scratch reimplementation,
 * not a wrapper, deliberately narrow (fixed numeric fields, no locale/timezone handling) rather
 * than a general strptime port, since that's all every current call site needs. Used
 * unconditionally on every platform (not just under #ifdef _WIN32) so there is exactly one
 * parsing implementation to verify instead of real strptime diverging from this replacement on
 * whatever edge case someone eventually hits.
 */
inline const char* strptime_utc(const char* s, char separator, std::tm* tm) {
    int year = 0, mon = 0, day = 0, hour = 0, min = 0, sec = 0, consumed = 0;
    char fmt[32];
    std::snprintf(fmt, sizeof(fmt), "%%d-%%d-%%d%c%%d:%%d:%%d%%n", separator);
    if (std::sscanf(s, fmt, &year, &mon, &day, &hour, &min, &sec, &consumed) != 6) {
        return nullptr;
    }
    tm->tm_year = year - 1900;
    tm->tm_mon = mon - 1;
    tm->tm_mday = day;
    tm->tm_hour = hour;
    tm->tm_min = min;
    tm->tm_sec = sec;
    tm->tm_isdst = 0;
    return s + consumed;
}

}  // namespace adai
