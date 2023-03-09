/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Save a structured n x n mesh of square elements on the unit square into a
   GLVis mesh file with the given name. */
#include <math.h>
void GLVis_PrintGlobalSquareMesh(const char *meshfile, int n)
{
   FILE *file;

   int Dim = 2;
   int NumOfVertices = (n + 1) * (n + 1);
   int NumOfElements = n * n;

   int i, j;
   double x, y;
   double h = 1.0 / n;

   if ((file = fopen(meshfile, "w")) == NULL)
   {
      printf("Error: can't open output file %s\n", meshfile);
      exit(1);
   }

   /* mesh header */
   fprintf(file, "MFEM mesh v1.0\n");
   fprintf(file, "\ndimension\n");
   fprintf(file, "%d\n", Dim);

   /* mesh elements */
   fprintf(file, "\nelements\n");
   fprintf(file, "%d\n", NumOfElements);
   for (j = 0; j < n; j++)
      for (i = 0; i < n; i++)
         fprintf(file, "1 3 %d %d %d %d\n", i + j * (n + 1), i + 1 + j * (n + 1),
                 i + 1 + (j + 1) * (n + 1), i + (j + 1) * (n + 1));

   /* boundary will be generated by GLVis */
   fprintf(file, "\nboundary\n");
   fprintf(file, "0\n");

   /* mesh vertices */
   fprintf(file, "\nvertices\n");
   fprintf(file, "%d\n", NumOfVertices);
   fprintf(file, "%d\n", Dim);
   for (j = 0; j < n + 1; j++)
      for (i = 0; i < n + 1; i++)
      {
         x = i * h;
         y = j * h;
         fprintf(file, "%.14e %.14e\n", x, y);
      }

   fflush(file);
   fclose(file);
}