//
//  timemeasure.h
//  GradientMethod
//
//  Created by MAAZ on 12/12/19.
//  Copyright © 2019 MAAZ. All rights reserved.
//

#ifndef timemeasure_h
#define timemeasure_h





#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#ifdef __APPLE__
#define CCPAL_INIT_LIB struct timespec tsi, tsf; \
double elaps_s; long elaps_ns; \
clock_serv_t cclock; \
mach_timespec_t mts;
#else
#define CCPAL_INIT_LIB struct timespec tsi, tsf; \
double elaps_s; long elaps_ns;
#endif

#ifdef __APPLE__
#define CCPAL_START_MEASURING \
host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock); \
clock_get_time(cclock, &mts); \
mach_port_deallocate(mach_task_self(), cclock); \
tsi.tv_sec = mts.tv_sec; \
tsi.tv_nsec = mts.tv_nsec;

#define CCPAL_STOP_MEASURING \
host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock); \
clock_get_time(cclock, &mts); \
mach_port_deallocate(mach_task_self(), cclock); \
tsf.tv_sec = mts.tv_sec; \
tsf.tv_nsec = mts.tv_nsec; \
elaps_s = difftime(tsf.tv_sec, tsi.tv_sec); \
elaps_ns = tsf.tv_nsec - tsi.tv_nsec;
#else
#ifdef CLOCK_PROCESS_CPUTIME_ID
/* cpu time in the current process */

#define CCPAL_START_MEASURING clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tsi);

#define CCPAL_STOP_MEASURING clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tsf); \
elaps_s = difftime(tsf.tv_sec, tsi.tv_sec); \
elaps_ns = tsf.tv_nsec - tsi.tv_nsec;

#else

/* this one should be appropriate to avoid errors on multiprocessors systems */

#define CCPAL_START_MEASURING clock_gettime(CLOCK_MONOTONIC_RAW, &tsi);

#define CCPAL_STOP_MEASURING clock_gettime(CLOCK_MONOTONIC_RAW, &tsf); \
elaps_s = difftime(tsf.tv_sec, tsi.tv_sec); \
elaps_ns = tsf.tv_nsec - tsi.tv_nsec;

#endif

#endif

#define CCPAL_REPORT_ANALYSIS fprintf (stdout, "We have spent %lf seconds executing previous code section.\n", elaps_s + ((double)elaps_ns) / 1.0e9 );





#endif /* timemeasure_h */
