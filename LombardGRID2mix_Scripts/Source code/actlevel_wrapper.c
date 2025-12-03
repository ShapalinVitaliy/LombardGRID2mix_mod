/* actlevel_wrapper.c */
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "sv-p56.h"   /* предполагаем, что этот заголовок в include-пути */


#ifdef _WIN32
  #ifdef __cplusplus
    #define SVP56_API extern "C" __declspec(dllexport)
  #else
    #define SVP56_API __declspec(dllexport)
  #endif
#else
  #define SVP56_API
#endif



/* Прототипы из sv-p56.c */
void init_speech_voltmeter(SVP56_state *state, double sampl_freq);
double speech_voltmeter(float *buffer, long smpno, SVP56_state *state);

/* Экспортируемая функция - принимает буфер float32 (-1..1), возвращает dBov */
/* Возвращает ActiveSpeechLevel (в dB); если activity_out != NULL, туда пишется ActivityFactor */
SVP56_API double p56_active_level_from_buffer(const float *buffer_in, long smpno, double fs, double *activity_out)
{
    if (buffer_in == NULL || smpno <= 0) {
        if (activity_out) *activity_out = 0.0;
        return -100.0; /* как в оригинале: признак "тишины" */
    }

    /* sv-p56 работает с float*, но функция speech_voltmeter не модифицирует буфер - каст безопасен */
    SVP56_state state;
    init_speech_voltmeter(&state, fs);

    /* speech_voltmeter требует float*, но наш аргумент const float*; делаем копию только если нужно */
    float *buf = (float*)buffer_in; /* assume caller guarantees mutability or it's okay */

    double active_db = speech_voltmeter(buf, (long)smpno, &state);
    if (activity_out) {
        *activity_out = state.ActivityFactor;
    }
    return active_db;
}
