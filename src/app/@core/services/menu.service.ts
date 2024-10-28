// src/app/@core/services/menu.service.ts
import { Injectable } from '@angular/core';
import { MENU_ITEMS } from '../../pages/pages-menu'; 

@Injectable({
  providedIn: 'root'
})
export class MenuService {
  private menuItems = MENU_ITEMS;

  constructor() { console.log('MenuService instantiated'); }

  searchMenuItems(query: string) {
    const results = this.menuItems.filter(item =>
      item.title.toLowerCase().includes(query.toLowerCase())
    );
    console.log('Filtered results:', results); // Ensure this logs and is correct
    return results;
  }
  
}
