#!/usr/bin/env python3
"""Locked exp224-style HMM FB returning smoothed + filtered position marginals."""
from __future__ import annotations

import numpy as np
from numba import njit

@njit(cache=True, nogil=True)
def _hmm2_fb_filt_smooth(em, dm, dz, sp, rates, sig_r, sig_p, start_p, start_sig, r0, r0_sig, lam, mom):
    """Same locked exp224 FB as exact_hmm_posterior_source._hmm2_fb, plus filtered marginal."""
    T, P = em.shape
    R = len(rates)
    sr = rates[1] - rates[0] if R > 1 else 1.0
    NEG = np.float32(-1e18)
    alpha = np.full((T, P, R), NEG, np.float32)
    prev = np.full((P, R), NEG, np.float32)
    for p in range(P):
        dpos = (p - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r in range(R):
            dr = (rates[r] - r0) / r0_sig
            prev[p, r] = np.float32(lp0 - 0.5 * dr * dr)
    tmp = np.empty((P, R), np.float32)
    cur = np.empty((P, R), np.float32)
    for t in range(T):
        sgr = sig_r * np.sqrt(dm[t])
        v_r = (sgr / sr) ** 2
        lrk = np.empty((R, 3), np.float64)
        for r in range(R):
            m_r = -(1.0 - mom) * rates[r] * dm[t] / sr
            pp = 0.5 * (v_r + m_r)
            pm = 0.5 * (v_r - m_r)
            if pp < 1e-12:
                pp = 1e-12
            if pm < 1e-12:
                pm = 1e-12
            tot = pp + pm
            if tot > 0.9:
                pp *= 0.9 / tot
                pm *= 0.9 / tot
            lrk[r, 0] = np.log(pm)
            lrk[r, 1] = np.log(1.0 - pp - pm)
            lrk[r, 2] = np.log(pp)
        for p in range(P):
            for r2 in range(R):
                m2 = NEG
                k0 = r2 - 1 if r2 - 1 >= 0 else 0
                k1 = r2 + 1 if r2 + 1 <= R - 1 else R - 1
                for r in range(k0, k1 + 1):
                    v = prev[p, r] + lrk[r, r2 - r + 1]
                    if v > m2:
                        m2 = v
                if m2 > NEG / 2:
                    ss = 0.0
                    for r in range(k0, k1 + 1):
                        ss += np.exp(prev[p, r] + lrk[r, r2 - r + 1] - m2)
                    tmp[p, r2] = np.float32(m2 + np.log(ss))
                else:
                    tmp[p, r2] = NEG
        spe = sig_p if sig_p > 0.35 * sp else 0.35 * sp
        for r2 in range(R):
            mu = rates[r2] * dm[t] - dz[t]
            b0 = int(np.floor(mu / sp + 0.5))
            lp = np.empty(5, np.float64)
            for k in range(5):
                d = (b0 - 2 + k) * sp - mu
                lp[k] = -0.5 * (d / spe) ** 2
            mx = lp[0]
            for k in range(1, 5):
                if lp[k] > mx:
                    mx = lp[k]
            ss = 0.0
            for k in range(5):
                ss += np.exp(lp[k] - mx)
            lz = mx + np.log(ss)
            for k in range(5):
                lp[k] -= lz
            for p2 in range(P):
                m2 = NEG
                for k in range(5):
                    p1 = p2 - (b0 - 2 + k)
                    if p1 < 0 or p1 >= P:
                        continue
                    v = tmp[p1, r2] + lp[k]
                    if v > m2:
                        m2 = v
                if m2 > NEG / 2:
                    ss2 = 0.0
                    for k in range(5):
                        p1 = p2 - (b0 - 2 + k)
                        if p1 < 0 or p1 >= P:
                            continue
                        ss2 += np.exp(tmp[p1, r2] + lp[k] - m2)
                    cur[p2, r2] = np.float32(m2 + np.log(ss2) + lam * em[t, p2])
                else:
                    cur[p2, r2] = NEG
        for p in range(P):
            for r in range(R):
                alpha[t, p, r] = cur[p, r]
                prev[p, r] = cur[p, r]

    filt_p = np.zeros((T, P), np.float64)
    for t in range(T):
        mx = NEG
        for p in range(P):
            for r in range(R):
                if alpha[t, p, r] > mx:
                    mx = alpha[t, p, r]
        total = 0.0
        for p in range(P):
            acc = 0.0
            for r in range(R):
                acc += np.exp(alpha[t, p, r] - mx)
            filt_p[t, p] = acc
            total += acc
        if total > 0.0:
            for p in range(P):
                filt_p[t, p] /= total

    mx = NEG
    for p in range(P):
        for r in range(R):
            if alpha[T - 1, p, r] > mx:
                mx = alpha[T - 1, p, r]
    ss = 0.0
    for p in range(P):
        for r in range(R):
            ss += np.exp(alpha[T - 1, p, r] - mx)
    loglik = float(mx) + np.log(ss)

    post_p = np.zeros((T, P), np.float64)
    beta_next = np.zeros((P, R), np.float32)
    beta_cur = np.empty((P, R), np.float32)
    beta_tmp = np.empty((P, R), np.float32)
    for t in range(T - 1, -1, -1):
        mxp = NEG
        for p in range(P):
            for r in range(R):
                v = alpha[t, p, r] + beta_next[p, r]
                if v > mxp:
                    mxp = v
        total = 0.0
        for p in range(P):
            acc = 0.0
            for r in range(R):
                acc += np.exp(alpha[t, p, r] + beta_next[p, r] - mxp)
            post_p[t, p] = acc
            total += acc
        if total > 0.0:
            for p in range(P):
                post_p[t, p] /= total
        if t == 0:
            break
        sgr = sig_r * np.sqrt(dm[t])
        v_r = (sgr / sr) ** 2
        lrk = np.empty((R, 3), np.float64)
        for r in range(R):
            m_r = -(1.0 - mom) * rates[r] * dm[t] / sr
            pp = 0.5 * (v_r + m_r)
            pm = 0.5 * (v_r - m_r)
            if pp < 1e-12:
                pp = 1e-12
            if pm < 1e-12:
                pm = 1e-12
            tot = pp + pm
            if tot > 0.9:
                pp *= 0.9 / tot
                pm *= 0.9 / tot
            lrk[r, 0] = np.log(pm)
            lrk[r, 1] = np.log(1.0 - pp - pm)
            lrk[r, 2] = np.log(pp)
        spe = sig_p if sig_p > 0.35 * sp else 0.35 * sp
        for r2 in range(R):
            mu = rates[r2] * dm[t] - dz[t]
            b0 = int(np.floor(mu / sp + 0.5))
            lp = np.empty(5, np.float64)
            for k in range(5):
                d = (b0 - 2 + k) * sp - mu
                lp[k] = -0.5 * (d / spe) ** 2
            mx2 = lp[0]
            for k in range(1, 5):
                if lp[k] > mx2:
                    mx2 = lp[k]
            ss2 = 0.0
            for k in range(5):
                ss2 += np.exp(lp[k] - mx2)
            lz = mx2 + np.log(ss2)
            for k in range(5):
                lp[k] -= lz
            for p1 in range(P):
                m2 = NEG
                for k in range(5):
                    p2 = p1 + (b0 - 2 + k)
                    if p2 < 0 or p2 >= P:
                        continue
                    v = lp[k] + lam * em[t, p2] + beta_next[p2, r2]
                    if v > m2:
                        m2 = v
                if m2 > NEG / 2:
                    ss3 = 0.0
                    for k in range(5):
                        p2 = p1 + (b0 - 2 + k)
                        if p2 < 0 or p2 >= P:
                            continue
                        ss3 += np.exp(lp[k] + lam * em[t, p2] + beta_next[p2, r2] - m2)
                    beta_tmp[p1, r2] = np.float32(m2 + np.log(ss3))
                else:
                    beta_tmp[p1, r2] = NEG
        for p in range(P):
            for r in range(R):
                m2 = NEG
                k0 = r - 1 if r - 1 >= 0 else 0
                k1 = r + 1 if r + 1 <= R - 1 else R - 1
                for r2 in range(k0, k1 + 1):
                    v = lrk[r, r2 - r + 1] + beta_tmp[p, r2]
                    if v > m2:
                        m2 = v
                if m2 > NEG / 2:
                    ss4 = 0.0
                    for r2 in range(k0, k1 + 1):
                        ss4 += np.exp(lrk[r, r2 - r + 1] + beta_tmp[p, r2] - m2)
                    beta_cur[p, r] = np.float32(m2 + np.log(ss4))
                else:
                    beta_cur[p, r] = NEG
        mxn = NEG
        for p in range(P):
            for r in range(R):
                if beta_cur[p, r] > mxn:
                    mxn = beta_cur[p, r]
        ssn = 0.0
        for p in range(P):
            for r in range(R):
                ssn += np.exp(beta_cur[p, r] - mxn)
        norm = mxn + np.log(ssn) if ssn > 0.0 else 0.0
        for p in range(P):
            for r in range(R):
                beta_next[p, r] = np.float32(beta_cur[p, r] - norm)
    return post_p, filt_p, loglik


