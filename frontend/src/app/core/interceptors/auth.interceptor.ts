import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';

import { AuthService } from '../services/auth.service';

const ADMIN_PATHS = ['/retrain', '/admin', '/registry/promote', '/monitoring/drift/auto-retrain'];

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  const jwt = auth.snapshot().jwt;
  if (!jwt) {
    return next(req);
  }
  const needsBearer = ADMIN_PATHS.some((p) => req.url.includes(p));
  if (!needsBearer) {
    return next(req);
  }
  return next(
    req.clone({
      setHeaders: { Authorization: `Bearer ${jwt}` },
    }),
  );
};
