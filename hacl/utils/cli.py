#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['wait_key']


def wait_key(cv=True):
    if cv:
        print('Press any key on the OpenCV window to continue...')
        import cv2

        key = cv2.waitKey(0)
        if ord('q') == key:
            raise KeyboardInterrupt
    else:
        input('Press <Enter> to continue...')
