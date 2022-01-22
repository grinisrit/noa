// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>

namespace noaTNL {
namespace Meshes {
namespace DistributedMeshes {

//index of direction can be written as number in 3-base system
//  -> 1 order x axis, 2 order y axis, 3 order z axis
//  -> 0 - not used, 1 negative direction, 2 positive direction
//finaly we subtrackt 1 because we dont need (0,0,0) aka 0 aka no direction

//enum Directions2D { Left = 0 , Right = 1 , Up = 2, UpLeft =3, UpRight=4, Down=5, DownLeft=6, DownRight=7 };

/*MEH - osa zed je zdola nahoru, asi---
enum Directions3D { West = 0 , East = 1 ,
                    North = 2, NorthWest = 3, NorthEast = 4,
                    South = 5, SouthWest = 6, SouthEast = 7,
                    Top = 8, TopWest = 9, TopEast =10,
                    TopNorth = 11, TopNorthWest = 12, TopNorthEast = 13,
                    TopSouth = 14, TopSouthWest = 15,TopSouthEast = 16,
                    Bottom = 17 ,BottomWest = 18 , BottomEast = 19 ,
                    BottomNorth = 20, BottomNorthWest = 21, BottomNorthEast = 22,
                    BottomSouth = 23, BottomSouthWest = 24, BottomSouthEast = 25,
                  };*/

/*
with self
enum Directions3D {
                    ZzYzXz =  0, ZzYzXm =  1, ZzYzXp =  2,
                    ZzYmXz =  3, ZzYmXm =  4, ZzYmXp =  5,
                    ZzYpXz =  6, ZzYpXm =  7, ZzYpXp =  8,
                    ZmYzXz =  9, ZmYzXm = 10, ZmYzXp = 11,
                    ZmYmXz = 12, ZmYmXm = 13, ZmYmXp = 14,
                    ZmYpXz = 15, ZmYpXm = 16, ZmYpXp = 17,
                    ZpYzXz = 18, ZpYzXm = 19, ZpYzXp = 20,
                    ZpYmXz = 21, ZpYmXm = 22, ZpYmXp = 23,
                    ZpYpXz = 24, ZpYpXm = 25, ZpYpXp = 26
                  };
*/

enum Directions3D {
                    ZzYzXm =  0, ZzYzXp =  1,
                    ZzYmXz =  2, ZzYmXm =  3, ZzYmXp =  4,
                    ZzYpXz =  5, ZzYpXm =  6, ZzYpXp =  7,
                    ZmYzXz =  8, ZmYzXm =  9, ZmYzXp = 10,
                    ZmYmXz = 11, ZmYmXm = 12, ZmYmXp = 13,
                    ZmYpXz = 14, ZmYpXm = 15, ZmYpXp = 16,
                    ZpYzXz = 17, ZpYzXm = 18, ZpYzXp = 19,
                    ZpYmXz = 20, ZpYmXm = 21, ZpYmXp = 22,
                    ZpYpXz = 23, ZpYpXm = 24, ZpYpXp = 25
                  };


class Directions {

public:
    template<int numerofDriection>
    static int getDirection(Containers::StaticVector<numerofDriection,int> directions) //takes +/- nuber of ax (i.e. (-2,+3))
    {
        int result=0;
        for(int i=0;i<directions.getSize();i++)
            result+=add(directions[i]);
        return result-1;
    }

    template<int dim>
    static Containers::StaticVector<dim,int> getXYZ(int neighbor)// return neighbor as direction like (0,-1,1)
    {
        Containers::StaticVector<dim,int> res;
        int number=neighbor+1;
        for(int i=0;i<dim;i++)
        {
            int direction=number%3;
            if(direction==0)
                res[i]=0;
            if(direction==1)
                res[i]=-1;
            if(direction==2)
                res[i]=1;
            number=number/3;
        }
        return res;
    }


 /*   static int getDirection(int direction)
    {
        int result=0;
        result+=add(direction);
        return result-1;
    }

    static int getDirection(int direction1,int direction2)
    {
        int result=0;
        result+=add(direction1);
        result+=add(direction2);
        return result-1;
    }

    static int getDirection(int direction1,int direction2, int direction3)
    {
        int result=0;
        result+=add(direction1);
        result+=add(direction2);
        result+=add(direction3);
        return result-1;
    }*/

    static constexpr int add(int direction)
    {
        if(direction==0)
            return 0;

        if(direction>0)
            return 2*i3pow(direction-1); //positive direction has higer index
        else
            return i3pow(-direction-1);
    }

    // return 3^exp
    static constexpr int i3pow(int exp)
    {
        int ret=1;
        for(int i=0;i<exp;i++)
            ret*=3;
        return ret;
    }
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace noaTNL
