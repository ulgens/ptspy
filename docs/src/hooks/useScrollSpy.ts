import { useEffect, useRef, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";

/**
 * Custom hook that updates the URL based on scroll position.
 * Uses Intersection Observer to detect which section is closest to the top.
 */
export function useScrollSpy(
  packageId: string | undefined,
  initialMemberId?: string
) {
  const navigate = useNavigate();
  const location = useLocation();
  const observerRef = useRef<IntersectionObserver | null>(null);
  // Start paused if there's an initial memberId to scroll to
  const isUserNavigatingRef = useRef(!!initialMemberId);
  const currentMemberRef = useRef<string | null>(initialMemberId || null);

  // Debounce URL updates to avoid excessive navigation
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const updateURL = useCallback(
    (memberId: string) => {
      if (!packageId || isUserNavigatingRef.current) return;

      // Avoid updating if we're already at this URL
      const targetPath = `/docs/${packageId}/${memberId}`;
      if (location.pathname === targetPath) return;

      // Clear any pending update
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // Debounce the URL update
      timeoutRef.current = setTimeout(() => {
        // Use replaceState to update URL without adding to history
        // This prevents back button from going through every scrolled section
        navigate(targetPath, { replace: true });
      }, 100);
    },
    [packageId, navigate, location.pathname]
  );

  useEffect(() => {
    if (!packageId || packageId === "typing") return;

    // Longer delay to ensure DOM elements are rendered AND initial scroll has completed
    const setupDelay = setTimeout(() => {
      // Reset the navigation flag after initial scroll would have completed
      isUserNavigatingRef.current = false;
      // Find all content sections with IDs
      // This includes: functions (direct children), classes, and class methods (nested deeper)
      const sections = document.querySelectorAll<HTMLElement>(".prose [id]");

      if (sections.length === 0) return;

      // Track which sections are intersecting and their positions
      const visibleSections = new Map<string, number>();

      observerRef.current = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            const id = entry.target.id;
            if (!id) return;

            if (entry.isIntersecting) {
              // Store the top position of this section
              visibleSections.set(id, entry.boundingClientRect.top);
            } else {
              visibleSections.delete(id);
            }
          });

          // Find the section closest to the top (but still visible)
          let closestId: string | null = null;
          let closestDistance = Infinity;

          visibleSections.forEach((top, id) => {
            // We want the section whose top is closest to the viewport top
            // but still visible (top can be negative if partially scrolled past)
            const distance = Math.abs(top);
            if (distance < closestDistance) {
              closestDistance = distance;
              closestId = id;
            }
          });

          // Update URL if we found a visible section
          if (closestId && closestId !== currentMemberRef.current) {
            currentMemberRef.current = closestId;
            updateURL(closestId);
          }
        },
        {
          // Root margin: observe when sections enter the top portion of viewport
          rootMargin: "-80px 0px -70% 0px",
          threshold: [0, 0.1, 0.5, 1],
        }
      );

      // Observe all sections
      sections.forEach((section) => {
        observerRef.current?.observe(section);
      });
    }, 500);

    return () => {
      clearTimeout(setupDelay);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      observerRef.current?.disconnect();
    };
  }, [packageId, updateURL]);

  // Method to mark that user is navigating (e.g., clicking menu)
  // This pauses scroll spy AND signals that we should scroll to the target
  const setUserNavigating = useCallback(() => {
    isUserNavigatingRef.current = true;
    setTimeout(() => {
      isUserNavigatingRef.current = false;
    }, 500);
  }, []);

  // Check if current navigation is user-initiated
  const isUserNavigating = useCallback(() => {
    return isUserNavigatingRef.current;
  }, []);

  return { setUserNavigating, isUserNavigating };
}
