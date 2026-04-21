import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

import { AuthService } from '../services/auth.service';
import { ToastService } from '../services/toast.service';

export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  const toast = inject(ToastService);
  const auth = inject(AuthService);
  const router = inject(Router);

  return next(req).pipe(
    catchError((err: HttpErrorResponse) => {
      const detail =
        (err.error && (err.error.detail || err.error.message)) ||
        err.message ||
        'Request failed.';

      if (err.status === 0) {
        toast.error('Network unreachable', 'The API is not responding. Check your connection.');
      } else if (err.status === 401) {
        if (req.url.includes('/auth/token')) {
          toast.error('Sign-in failed', detail);
        } else {
          toast.warning('Session expired', 'Please sign in again to continue.');
          auth.logout();
          router.navigate(['/login']);
        }
      } else if (err.status === 403) {
        toast.error('Forbidden', detail);
      } else if (err.status === 422) {
        toast.warning('Validation error', detail);
      } else if (err.status >= 500) {
        toast.error('Server error', detail);
      } else if (err.status >= 400) {
        toast.warning(`Request rejected (${err.status})`, detail);
      }
      return throwError(() => err);
    }),
  );
};
